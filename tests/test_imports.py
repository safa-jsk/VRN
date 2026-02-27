#!/usr/bin/env python3
"""Test that all key modules are importable."""
import importlib
import unittest


MODULES = [
    "src.vrn",
    "src.vrn.config",
    "src.vrn.io",
    "src.vrn.perf",
    "src.vrn.metrics",
    "src.vrn.utils",
    "src.designB",
    "src.designB.io",
    "src.designB.pipeline",
    "src.designB.benchmark",
    "src.designB.convert_raw_to_npy",
    "src.designC",
    "src.designC.infer_facescape",
    "src.designC.eval_facescape",
]


class TestImports(unittest.TestCase):
    pass


def _make_test(mod_name):
    def test(self):
        importlib.import_module(mod_name)
    test.__name__ = f"test_import_{mod_name.replace('.', '_')}"
    return test


for _mod in MODULES:
    setattr(TestImports, f"test_import_{_mod.replace('.', '_')}", _make_test(_mod))


if __name__ == "__main__":
    unittest.main()
