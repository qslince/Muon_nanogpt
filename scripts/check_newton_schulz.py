"""Inspect the finite-step Newton–Schulz update used in modded-nanogpt.

Run from the repository root:
    python scripts/check_newton_schulz.py --output-dir results/ns_sanity

Only the original function is extracted with AST; the training script is never
imported or run. No torch.compile or training is used in this diagnostic.
"""

import argparse
import ast
import csv
import hashlib
import json
import platform
from pathlib import Path

import torch


UPSTREAM_COMMIT = "d6a58a321d459a9267243aecc3e0615cffc8f2fc"
DTYPES = {"fp64": torch.float64, "fp32": torch.float32,
          "bf16": torch.bfloat16, "fp16": torch.float16}
CASES = ("gaussian", "flat", "ill_conditioned", "rank_deficient", "zero")


def newton_schulz(matrix, steps, dtype, eps=1e-7):
    """Original arithmetic and normalization order, with explicit compute dtype.

    bf16 matches the original eager function. Other dtypes are diagnostic
    variants. Return the result before Muon's shape-dependent update scaling.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.to(dtype=dtype, copy=True)
    x /= x.norm() + eps
    if matrix.size(0) > matrix.size(1):
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = aa @ x
        x = a * x + b * bb + c * aa @ bb
    if matrix.size(0) > matrix.size(1):
        x = x.T
    return x


def load_original(path):
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "zeropower_via_newtonschulz5")
    function.decorator_list = []  # Inspect the eager algorithm, not compilation.
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function.name], hashlib.sha256(path.read_bytes()).hexdigest()


def make_matrix(rows, cols, case, generator):
    if case == "zero":
        return torch.zeros(rows, cols, dtype=torch.float64)
    if case == "gaussian":
        return torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    rank = min(rows, cols)
    left = torch.linalg.qr(torch.randn(rows, rank, generator=generator,
                                      dtype=torch.float64)).Q
    right = torch.linalg.qr(torch.randn(cols, rank, generator=generator,
                                       dtype=torch.float64)).Q
    spectrum = torch.ones(rank, dtype=torch.float64)
    if case == "ill_conditioned":
        spectrum = torch.logspace(0, -8, rank, dtype=torch.float64)
    elif case == "rank_deficient":
        spectrum[rank // 2:] = 0
    return (left * spectrum) @ right.T


def relative_error(value, reference):
    denominator = reference.norm().item()
    error = (value - reference).norm().item()
    return error / denominator if denominator else error


def parse_shape(value):
    try:
        rows, cols = map(int, value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use ROWSxCOLS, for example 16x32") from exc
    if min(rows, cols) < 2:
        raise argparse.ArgumentTypeError("Each dimension must be at least 2")
    return rows, cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1]
                        / "source_code/train_gpt2.py")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for CSV and JSON; existing files are not overwritten")
    parser.add_argument("--shapes", nargs="+", type=parse_shape,
                        default=[(16, 16), (16, 32), (32, 16), (64, 64)])
    parser.add_argument("--steps", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES,
                        default=["fp64", "fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps) < 0 or args.threads < 1:
        parser.error("Steps must be nonnegative; threads must be positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; select a new directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    original, source_sha256 = load_original(args.source)
    generator = torch.Generator().manual_seed(args.seed)
    records, spectra, errors = [], [], []
    oracle_checks = original_checks = 0

    with torch.no_grad():
        for rows, cols in args.shapes:
            for case in CASES:
                matrix = make_matrix(rows, cols, case, generator).to(args.device)
                u, singular, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
                tolerance = max(rows, cols) * torch.finfo(torch.float64).eps * singular.max()
                active = singular > tolerance
                partial_polar = (u[:, active] @ vh[active]).to(args.device)
                for steps in args.steps:
                    reference = newton_schulz(matrix, steps, torch.float64)
                    # Independent spectral oracle for the finite polynomial.
                    values = singular / (matrix.cpu().norm() + 1e-7)
                    for _ in range(steps):
                        values = 3.4445 * values - 4.7750 * values**3 + 2.0315 * values**5
                    oracle = ((u * values) @ vh).to(args.device)
                    torch.testing.assert_close(reference, oracle, rtol=1e-9, atol=1e-10)
                    oracle_checks += 1
                    for name in args.dtypes:
                        try:
                            result = newton_schulz(matrix, steps, DTYPES[name])
                            assert result.shape == matrix.shape, "Output shape changed"
                            if not torch.isfinite(result).all():
                                raise ValueError("Non-finite output")
                            if case == "zero":
                                assert torch.count_nonzero(result) == 0, "Zero input changed"
                            if name == "bf16":
                                expected = original(matrix.clone(), steps=steps)
                                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                                original_checks += 1
                            output = result.double()
                            output_singular = torch.linalg.svdvals(output.cpu())
                            records.append({
                                "case": case, "rows": rows, "cols": cols,
                                "steps": steps, "dtype": name,
                                "input_rank_fp64": int(active.sum()),
                                "sigma_min": output_singular.min().item(),
                                "sigma_max": output_singular.max().item(),
                                "relative_error_to_fp64": relative_error(output, reference),
                                "relative_error_to_partial_polar": relative_error(output, partial_polar),
                            })
                            spectra.append({"case": case, "shape": [rows, cols],
                                            "steps": steps, "dtype": name,
                                            "input_singular_values": singular.tolist(),
                                            "output_singular_values": output_singular.tolist()})
                        except (RuntimeError, ValueError, AssertionError) as exc:
                            errors.append({"case": case, "shape": [rows, cols],
                                           "steps": steps, "dtype": name, "error": str(exc)})

    args.output_dir.mkdir(parents=True)
    if records:
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    report = {
        "upstream_commit": UPSTREAM_COMMIT,
        "source_path": str(args.source.resolve()), "source_sha256": source_sha256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "seed": args.seed, "threads": args.threads,
        "shapes": args.shapes, "steps": args.steps, "dtypes": args.dtypes,
        "eps": 1e-7, "torch_compile": False,
        "spectral_oracle_checks": oracle_checks,
        "original_bf16_equivalence_checks": original_checks,
        "successful_cases": len(records), "errors": errors, "spectra": spectra,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Successful cases: {len(records)}; errors: {len(errors)}")
    print(f"FP64 spectral checks: {oracle_checks}; original BF16 checks: {original_checks}")
    print(f"Results: {args.output_dir.resolve()}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
