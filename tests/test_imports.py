"""The shared package and CLI help must work without either framework."""

import subprocess
import sys


def test_shared_imports_do_not_load_backends():
    script = """
import importlib.abc
import sys
class RejectFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'jax', 'optax'}:
            raise AssertionError(f'unexpected framework import: {fullname}')
sys.meta_path.insert(0, RejectFrameworks())
import huo
from huo.geometry import compute_ray_coordinates
from huo.cli import build_parser
geometry = huo.FanbeamGeometry(8, 32, 16, 12, 1.3)
assert compute_ray_coordinates(geometry)[0].shape == (17, 12)
assert '--backend' in build_parser().format_help()
assert huo.__all__ == ['FanbeamGeometry']
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
