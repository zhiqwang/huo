"""Measure CPU operator timings and retained autograd storage.

Run from the repository root with ``uv run python -m benchmarks.autograd``.
Saved storage counts unique live tensor buffers, not process peak memory.
"""

import argparse
import json
import math
import weakref

import torch
from torch.utils.benchmark import Timer

from huo import RadonFanbeam


def retained_bytes(references):
    storage = {}
    for reference in references:
        tensor = reference()
        if tensor is not None:
            buffer = tensor.untyped_storage()
            storage[buffer.data_ptr()] = buffer.nbytes()
    return sum(storage.values())


def measure(scanner, method, create_graph, threads, min_run_time):
    n, m, views = scanner.geometry.resolution, scanner.geometry.det_count, len(scanner.angles)
    shape = (n, n) if method == "forward" else (views, m)
    data = torch.rand(shape, generator=torch.Generator().manual_seed(42)).requires_grad_()
    operator = getattr(scanner, method)
    references = []

    def pack(tensor):
        saved = tensor.detach()
        references.append(weakref.ref(saved))
        return saved

    def evaluate():
        if method == "art":
            torch.manual_seed(0)
        loss = 0.5 * operator(data).square().sum()
        return torch.autograd.grad(loss, data, create_graph=create_graph)[0]

    if method == "art":
        torch.manual_seed(0)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = 0.5 * operator(data).square().sum()
        forward_bytes = retained_bytes(references)
        (gradient,) = torch.autograd.grad(loss, data, create_graph=create_graph)
        gradient_bytes = retained_bytes(references)
    del loss, gradient
    timing = Timer("evaluate()", globals={"evaluate": evaluate}, num_threads=threads).blocked_autorange(
        min_run_time=min_run_time
    )
    return {
        "method": method,
        "create_graph": create_graph,
        "forward_and_gradient_ms": timing.median * 1000,
        "iqr_ms": timing.iqr * 1000,
        "saved_after_forward_bytes": forward_bytes,
        "saved_after_gradient_bytes": gradient_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--det-count", type=int, default=160)
    parser.add_argument("--views", type=int, default=48)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.5)
    args = parser.parse_args()
    if args.views < 1 or args.threads < 1 or args.min_run_time <= 0:
        parser.error("views, threads, and min-run-time must be positive")
    torch.set_num_threads(args.threads)
    scanner = RadonFanbeam(
        args.resolution,
        [index * 2 * math.pi / args.views for index in range(args.views)],
        4 * args.resolution,
        2 * args.resolution,
        args.det_count,
        1.3,
    )
    results = [
        measure(scanner, method, create_graph, args.threads, args.min_run_time)
        for method in ("forward", "adjoint", "backprojection", "art")
        for create_graph in (False, True)
    ]
    print(json.dumps({"torch": torch.__version__, **vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
