from __future__ import annotations

import unittest

import numpy as np

from rf.two_port import SParameters, evaluate_match_efficiency, s_to_z, z_to_abcd


class TwoPortTests(unittest.TestCase):
    def test_identity_network_conversion(self) -> None:
        s = SParameters(s11=0j, s22=0j, s21=0.99 + 0j, s12=0.99 + 0j, zref_ohm=50.0)
        z = s_to_z(s)
        self.assertEqual(z.shape, (2, 2))
        abcd = z_to_abcd(z)
        self.assertTrue(np.all(np.isfinite(abcd)))

    def test_ideal_network_efficiency_near_one(self) -> None:
        s = SParameters(s11=0j, s22=0j, s21=1 + 0j, s12=1 + 0j, zref_ohm=50.0)
        p_in, p_out, eta = evaluate_match_efficiency(s, 50 + 0j, 50 + 0j)
        self.assertGreater(p_in, 0.0)
        self.assertAlmostEqual(eta, 1.0, places=6)
        self.assertAlmostEqual(p_in, p_out, places=6)


if __name__ == "__main__":
    unittest.main()
