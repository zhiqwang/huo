"""Compare CPU JAX prototypes with the current PyTorch API in fresh processes.

Run: uv run --all-extras python -m benchmarks.compare_jax --output results.json
Each worker is pinned to the same CPU, executes one backend/case/operation,
and synchronizes every timed result. Compilation and process RSS are reported
separately from warm timings. No timing thresholds are used as tests.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

CASES = {
    "small": (32, 40, 24, 1),
    "medium": (128, 160, 48, 1),
    "large": (256, 320, 90, 1),
    "batch": (128, 160, 48, 4),
}
BACKENDS = ("torch", "jax-vmap", "jax-scan", "jax-custom_jvp")
OPERATIONS = ("forward", "loss_grad", "jvp", "hvp", "reconstruct")


def pin_cpu(cpu):
    if cpu not in os.sched_getaffinity(0):
        raise ValueError(f"CPU {cpu} is outside this process's allowed affinity")
    os.sched_setaffinity(0, {cpu})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"


def rss_mib(field="VmRSS"):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("This benchmark requires Linux /proc RSS accounting")


def timing(call, synchronize, repeats, minimum_seconds):
    start = perf_counter()
    output = synchronize(call())
    first_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    for _ in range(2):
        output = synchronize(call())
    warm_ms = (perf_counter() - start) * 500
    count = max(1, min(200, math.ceil(minimum_seconds * 1000 / (repeats * max(warm_ms, 0.001)))))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(count):
            output = synchronize(call())
        samples.append((perf_counter() - start) * 1000 / count)
    return output, {
        "first_execution_ms": first_ms,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "calls_per_sample": count,
    }


def worker(args):
    import numpy as np

    data = np.load(args.data)
    n, m, views, batch = CASES[args.case]
    initialize_start = perf_counter()
    compile_ms = 0.0
    executable_memory = None
    if args.backend == "torch":
        import torch
        from huo import FanbeamGeometry
        from huo.torch import FanbeamProjector, reconstruct

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        project = FanbeamProjector(FanbeamGeometry(n, 4 * n, 2 * n, m, 1.3), data["angles"])
        image = torch.from_numpy(data["image"])
        measured = torch.from_numpy(data["measured"])
        if args.operation == "forward":
            call = lambda: project(image)
        elif args.operation == "loss_grad":
            image.requires_grad_()

            def call():
                loss = 0.5 * (project(image) - measured).square().mean()
                (gradient,) = torch.autograd.grad(loss, image)
                return loss.detach(), gradient

        elif args.operation in ("jvp", "hvp"):
            direction = torch.from_numpy(data["direction"])
            if args.operation == "jvp":
                call = lambda: torch.func.jvp(project, (image,), (direction,))
            else:
                gradient = torch.func.grad(lambda x: 0.5 * (project(x) - measured).square().mean())
                call = lambda: torch.func.jvp(gradient, (image,), (direction,))
        else:
            call = lambda: reconstruct(project, measured, steps=args.steps, lr=0.03)
        synchronize = lambda value: value
        as_array = lambda value: value.detach().numpy()
        versions = {"torch": torch.__version__, "numpy": np.__version__}
        device = "cpu"
        initialize_ms = (perf_counter() - initialize_start) * 1000
        baseline_rss = rss_mib()
    else:
        import jax
        import jax.numpy as jnp
        import jaxlib
        import optax
        from benchmarks.jax_projector import make_projector, make_reconstructor, projection_loss

        project = make_projector(implementation=args.backend.removeprefix("jax-"))
        image, measured, ray_x, ray_y, angles = (
            jnp.asarray(data[name]) for name in ("image", "measured", "ray_x", "ray_y", "angles")
        )
        if args.operation == "forward":
            function = project
            inputs = (image, ray_x, ray_y, angles)
        elif args.operation == "loss_grad":
            function = jax.value_and_grad(lambda x, y, rx, ry, a: projection_loss(project, x, y, rx, ry, a))
            inputs = (image, measured, ray_x, ray_y, angles)
        elif args.operation in ("jvp", "hvp"):

            def function(x, y, direction, rx, ry, a):
                if args.operation == "jvp":
                    operator = lambda value: project(value, rx, ry, a)
                else:
                    operator = jax.grad(lambda value: projection_loss(project, value, y, rx, ry, a))
                return jax.jvp(operator, (x,), (direction,))

            inputs = (image, measured, jnp.asarray(data["direction"]), ray_x, ray_y, angles)
        else:
            function = make_reconstructor(project, steps=args.steps, lr=0.03)
            inputs = (measured, jnp.zeros_like(image), ray_x, ray_y, angles)
        jax.block_until_ready(inputs)
        initialize_ms = (perf_counter() - initialize_start) * 1000
        baseline_rss = rss_mib()
        start = perf_counter()
        compiled = jax.jit(function).lower(*inputs).compile()
        compile_ms = (perf_counter() - start) * 1000
        analysis = compiled.memory_analysis()
        if analysis is not None:
            executable_memory = {
                name: getattr(analysis, name)
                for name in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            }
        call = lambda: compiled(*inputs)
        synchronize = jax.block_until_ready
        as_array = np.asarray
        versions = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
        }
        device = str(jax.devices()[0])

    rss_before_execution = rss_mib()
    repeats = min(args.repeats, 3) if args.operation == "reconstruct" else args.repeats
    output, times = timing(call, synchronize, repeats, args.minimum_seconds)
    result = {
        "case": args.case,
        "shape": [batch, n, n],
        "views": views,
        "detectors": m,
        "backend": args.backend,
        "operation": args.operation,
        "versions": versions,
        "device": device,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "initialization_ms": initialize_ms,
        "trace_compile_ms": compile_ms,
        **times,
        "cold_call_ms": compile_ms + times["first_execution_ms"],
        "rss_before_compile_mib": baseline_rss,
        "rss_before_execution_mib": rss_before_execution,
        # VmHWM belongs to the current executable's address space. ru_maxrss
        # can retain the parent's pre-exec high-water mark after fork/exec.
        "process_peak_rss_mib": rss_mib("VmHWM"),
        "xla_executable_memory": executable_memory,
    }
    arrays = output if isinstance(output, tuple) else (output,)
    np.savez(args.arrays, **{f"output_{index}": as_array(value) for index, value in enumerate(arrays)})
    print(json.dumps(result))


def compare(args):
    import numpy as np
    import torch
    from huo import FanbeamGeometry
    from huo.torch import FanbeamProjector

    torch.set_num_threads(1)
    cpu_model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_model,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "dtype": "float32",
        "steps": args.steps,
        "lr": 0.03,
        "repeats": args.repeats,
        "minimum_seconds": args.minimum_seconds,
        "load_average_start": os.getloadavg(),
        "method": "Fresh process per backend/case/operation; one shared CPU; explicit runtime geometry; synchronous wall time. RSS includes imports, compilation, executables, and arrays. XLA memory estimates are not RSS or PyTorch saved-tensor storage.",
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="huo-jax-") as directory:
        directory = Path(directory)
        for name in args.cases:
            n, m, views, batch = CASES[name]
            angles = np.arange(views, dtype=np.float32) * np.float32(2 * math.pi / views)
            project = FanbeamProjector(FanbeamGeometry(n, 4 * n, 2 * n, m, 1.3), angles)
            image = np.random.default_rng(42).normal(size=(batch, n, n)).astype(np.float32)
            direction = np.random.default_rng(43).normal(size=(batch, n, n)).astype(np.float32)
            phantom = torch.zeros(batch, n, n)
            phantom[:, n // 3 : 2 * n // 3, 3 * n // 8 : 5 * n // 8] = 1
            measured = project(phantom).numpy()
            data = directory / f"{name}.npz"
            np.savez(
                data,
                image=image,
                direction=direction,
                measured=measured,
                ray_x=project._ray_x.numpy(),
                ray_y=project._ray_y.numpy(),
                angles=angles,
            )
            for operation in args.operations:
                reference = None
                for backend in args.backends:
                    arrays_path = directory / "outputs.npz"
                    command = [
                        sys.executable,
                        "-m",
                        "benchmarks.compare_jax",
                        "--worker",
                        "--backend",
                        backend,
                        "--case",
                        name,
                        "--operation",
                        operation,
                        "--data",
                        str(data),
                        "--arrays",
                        str(arrays_path),
                        "--cpu",
                        str(args.cpu),
                        "--steps",
                        str(args.steps),
                        "--repeats",
                        str(args.repeats),
                        "--minimum-seconds",
                        str(args.minimum_seconds),
                    ]
                    process = subprocess.run(command, capture_output=True, text=True)
                    if process.returncode:
                        raise RuntimeError(f"{name}/{backend}/{operation}:\n{process.stderr}")
                    if process.stderr:
                        print(process.stderr, file=sys.stderr, end="")
                    result = json.loads(process.stdout)
                    with np.load(arrays_path) as saved:
                        arrays = [saved[key] for key in sorted(saved.files)]
                    if reference is None:
                        reference = arrays
                    errors = []
                    for actual, expected in zip(arrays, reference):
                        difference = actual.astype(np.float64) - expected.astype(np.float64)
                        error = {
                            "relative_l2": float(
                                np.linalg.norm(difference.ravel())
                                / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-30)
                            ),
                            "max_absolute": float(np.max(np.abs(difference))),
                        }
                        errors.append(error)
                        limit = 1e-3 if operation == "reconstruct" else 1e-4
                        if not np.isfinite(actual).all() or error["relative_l2"] > limit:
                            raise RuntimeError(
                                f"{name}/{backend}/{operation} failed numerical parity: {error}"
                            )
                    result["errors_vs_torch"] = errors
                    if operation == "reconstruct":
                        prediction = project(torch.from_numpy(arrays[0].copy()))
                        result["initial_loss"] = float(0.5 * np.mean(measured**2))
                        result["final_loss"] = float(
                            0.5 * (prediction - torch.from_numpy(measured)).square().mean()
                        )
                        if result["final_loss"] >= result["initial_loss"]:
                            raise RuntimeError("Reconstruction loss did not decrease")
                    report["results"].append(result)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        f"{name:6} {operation:11} {backend:14} {result['median_ms']:9.3f} ms; compile {result['trace_compile_ms']:8.1f} ms; peak RSS {result['process_peak_rss_mib']:7.1f} MiB",
                        flush=True,
                    )
    report["load_average_end"] = os.getloadavg()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/jax_cpu.json"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument(
        "--operations", nargs="+", choices=OPERATIONS, default=["forward", "loss_grad", "reconstruct"]
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-seconds", type=float, default=0.3)
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--operation", choices=OPERATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--data", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arrays", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 3 or args.minimum_seconds <= 0:
        parser.error("steps and minimum-seconds must be positive, repeats must be at least 3")
    if args.backends[0] != "torch":
        parser.error("The first backend must be torch, the numerical reference")
    pin_cpu(args.cpu)
    if args.worker:
        worker(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
