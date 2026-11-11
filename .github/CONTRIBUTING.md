# Contributing to huo

We want to make contributing to this project as easy and transparent as possible.

## TL;DR

We appreciate all contributions. If you are interested in contributing to `huo`, there are many ways to help out. Your contributions may fall into the following categories:

- It helps the project if you could

  - Report issues you're facing
  - Give a :+1: on issues that others reported and that are relevant to you

- Answering queries on the issue tracker, investigating bugs are very valuable contributions to the project.

- You would like to improve the documentation. This is no less important than improving the library itself! If you find a typo in the documentation, do not hesitate to submit a GitHub pull request.

- If you would like to fix a bug

  - please pick one from the [list of open issues labelled as "help wanted"](https://github.com/zhiqwang/huo/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
  - comment on the issue that you want to work on this issue
  - send a PR with your fix, see below.

- If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

## Development installation

### Install huo

Use Python 3.11 or newer and uv. Install both optional backends for development;
the base package depends only on NumPy. The `torch` extra uses the configured
PyTorch CPU index, and the `jax` extra supplies JAX and Optax:

```bash
git clone https://github.com/zhiqwang/huo.git
cd huo
uv sync --locked --all-extras --group test --python 3.12
source .venv/bin/activate
```

## Development Process

If you plan to modify the code or documentation, please follow the steps below:

1. Fork the repository and create your branch from `main`.
1. If you have modified the code (new feature or bug-fix), please add unit tests.
1. If you have changed APIs, update the documentation. Make sure the documentation builds.
1. Ensure the test suite passes.
1. Make sure your code passes the formatting checks.

### Code formatting

Contributions should be compatible with the supported Python versions and follow PEP 8.

### Unit tests

If you have modified the code by adding a new feature or a bug-fix, please add unit tests for that. To run a specific test:

```bash
uv run --locked --extra torch --group test pytest tests/test_autograd.py -q
```

If you would like to run all tests:

```bash
uv run --locked --all-extras --group test pytest tests -q
```

Run `npm --prefix packages ci` and `npm --prefix packages test` to build and
test the TypeScript operators before pytest; this enables Python/TypeScript
forward-projection parity checks. Reconstruction algorithms differ. CI runs
both suites on Python 3.11 and 3.12. It also installs non-editable base-only,
Torch-only, and JAX-only packages and runs `tools/check_installation.py` with
the installed environment's `python -I` to check dependency isolation and CLI
behavior. Backend tests skip when their optional dependencies are absent.

For CPU timing and retained autograd storage, run
`uv run --locked --extra torch python -m benchmarks.autograd`. The benchmark reports the
scanner dimensions and thread count with its results. Numerical and retained
storage checks run in CI; timings are for local comparison.

Run the public JAX contracts and the cross-backend numerical checks with
`uv run --locked --all-extras --group test pytest tests/test_jax.py tests/test_jax_reference.py -q`.
See `docs/jax_benchmark.md` for isolated CPU comparisons and the recorded
measurements. Do not run competing benchmarks concurrently on the selected CPU.

### Documentation

The Python API is shared NumPy geometry plus explicit `huo.torch` and `huo.jax`
backends, each exposing a forward projector and loss/optimizer workflow.
Keep the tutorial, API reference, CLI examples, and autograd derivation aligned
with that interface. The TypeScript ART example and its proofs are documented
separately. Follow `docs/publishing.md` for a strict Sphinx build.

### Pull Request Recommendations

If all previous checks (flake8, mypy, unit tests) are passing, please send a PR. Submitted PR will pass other tests on different operation systems, python versions and hardwares. To allow your work to be integrated as seamlessly as possible, we advise you to:

- :white_check_mark: Verify your PR is **up-to-date with upstream/main**. You could update your PR to upstream/main by running the following code, don't forget replacing 'feature' with the name of your local branch:

  ```bash
  git remote add upstream https://github.com/zhiqwang/huo.git
  git fetch upstream
  git rebase upstream/main
  git checkout -b feature  # <--- REPLACE 'feature' WITH YOUR LOCAL BRANCH NAME
  # ADD YOUR PROPOSED CHANGES HERE
  git add .
  git commit -m "YOUR REVISION MESSAGE"
  git push origin feature
  ```

- :white_check_mark: Verify all Continuous Integration (CI) **checks are passing**.

- :white_check_mark: Reduce changes to the absolute **minimum** required for your bug fix or feature addition. _"It is not daily increase but daily decrease, hack away the unessential. The closer to the source, the less wastage there is."_ — Bruce Lee

For more details about pull requests workflow, please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## License

By contributing to `huo`, you agree that your contributions will be licensed under the [LICENSE](LICENSE) file in the root directory of this source tree.
