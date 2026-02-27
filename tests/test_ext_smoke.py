#!/usr/bin/env python3
"""Smoke test for CUDA extensions (skip gracefully if not built)."""
import unittest


class TestExtSmoke(unittest.TestCase):
    def test_marching_cubes_ext_import(self):
        """Try to import marching_cubes_cuda_ext; skip if not built."""
        try:
            from external.marching_cubes_cuda_ext.cuda_marching_cubes import marching_cubes_gpu  # noqa: F401
        except ImportError:
            self.skipTest("marching_cubes_cuda_ext not built (run scripts/build_ext.sh)")

    def test_chamfer_ext_import(self):
        """Try to import chamfer extension; skip if not built."""
        try:
            import chamfer  # noqa: F401
        except ImportError:
            self.skipTest("chamfer_ext not built (run scripts/build_ext.sh)")


if __name__ == "__main__":
    unittest.main()
