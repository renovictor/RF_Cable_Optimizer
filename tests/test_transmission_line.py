from __future__ import annotations

import math
import unittest

from rf.transmission_line import (
    deembed_load_impedance,
    input_impedance,
    reflection_coefficient,
    return_loss_db,
    vswr,
)


class TransmissionLineTests(unittest.TestCase):
    def test_matched_lossless_line(self) -> None:
        z0 = 50 + 0j
        zl = 50 + 0j
        gamma = reflection_coefficient(zl, z0)
        self.assertAlmostEqual(abs(gamma), 0.0, places=12)
        self.assertAlmostEqual(vswr(gamma), 1.0, places=12)
        self.assertTrue(math.isinf(return_loss_db(gamma)))

    def test_forward_inverse_impedance_consistency(self) -> None:
        z0 = 40 + 0j
        zl = 3 + 19j
        gamma = 1j * 1.23
        length = 1.03
        zin = input_impedance(zl, z0, gamma, length)
        zl_back = deembed_load_impedance(zin, z0, gamma, length)
        self.assertAlmostEqual(zl_back.real, zl.real, places=9)
        self.assertAlmostEqual(zl_back.imag, zl.imag, places=9)


if __name__ == "__main__":
    unittest.main()

