#!/usr/bin/env python3
"""Smoke test – create a small synthetic volume and run baseline MC."""
import unittest
import numpy as np


class TestPipelineSmoke(unittest.TestCase):
    def test_baseline_marching_cubes(self):
        """Run CPU marching cubes on a tiny sphere volume."""
        from src.designB.pipeline import marching_cubes_baseline

        # Create a 32^3 sphere volume
        coords = np.mgrid[:32, :32, :32].astype(np.float32)
        centre = np.array([16, 16, 16]).reshape(3, 1, 1, 1)
        dist = np.sqrt(((coords - centre) ** 2).sum(axis=0))
        volume = (dist < 10).astype(np.float32)

        verts, faces = marching_cubes_baseline(volume, threshold=0.5)
        self.assertGreater(len(verts), 0, "Expected non-empty vertices")
        self.assertGreater(len(faces), 0, "Expected non-empty faces")
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)

    def test_config_defaults(self):
        """Verify config defaults match spec."""
        from src.vrn.config import load_config
        cfg = load_config()
        self.assertEqual(cfg["threshold"], 0.5)
        self.assertEqual(cfg["warmup_iters"], 15)

    def test_f1_score(self):
        """Basic F1 computation."""
        from src.vrn.metrics import f1_score
        score = f1_score(tp=80, fp=10, fn=10)
        self.assertAlmostEqual(score, 0.8, places=5)


if __name__ == "__main__":
    unittest.main()
