from __future__ import annotations

import unittest

import numpy as np

from rf.smith_math import gamma_from_impedance, impedance_from_gamma, normalized_impedance


class SmithMathTests(unittest.TestCase):
    def test_matched_load_maps_to_zero_gamma(self) -> None:
        z0 = 50.0 + 0j
        gamma = gamma_from_impedance(np.array([z0]), z0)[0]
        self.assertAlmostEqual(gamma.real, 0.0, places=12)
        self.assertAlmostEqual(gamma.imag, 0.0, places=12)

    def test_short_maps_to_minus_one_gamma(self) -> None:
        z0 = 50.0 + 0j
        gamma = gamma_from_impedance(np.array([0.0 + 0j]), z0)[0]
        self.assertAlmostEqual(gamma.real, -1.0, places=12)
        self.assertAlmostEqual(gamma.imag, 0.0, places=12)

    def test_open_approximation_maps_near_plus_one_gamma(self) -> None:
        z0 = 50.0 + 0j
        gamma = gamma_from_impedance(np.array([1e15 + 0j]), z0)[0]
        self.assertAlmostEqual(gamma.real, 1.0, places=10)
        self.assertAlmostEqual(gamma.imag, 0.0, places=10)

    def test_inductive_impedance_maps_to_upper_half(self) -> None:
        z0 = 50.0 + 0j
        gamma = gamma_from_impedance(np.array([z0 + 1j * z0]), z0)[0]
        self.assertGreater(gamma.imag, 0.0)

    def test_capacitive_impedance_maps_to_lower_half(self) -> None:
        z0 = 50.0 + 0j
        gamma = gamma_from_impedance(np.array([z0 - 1j * z0]), z0)[0]
        self.assertLess(gamma.imag, 0.0)

    def test_normalization(self) -> None:
        z0 = 25.0 + 0j
        z = np.array([25.0 + 25.0j], dtype=np.complex128)
        z_norm = normalized_impedance(z, z0)[0]
        self.assertAlmostEqual(z_norm.real, 1.0, places=12)
        self.assertAlmostEqual(z_norm.imag, 1.0, places=12)

    def test_round_trip_impedance_gamma_impedance(self) -> None:
        z0 = 50.0 + 0j
        z = np.array([30.0 + 15.0j, 80.0 - 20.0j, 10.0 + 40.0j], dtype=np.complex128)
        gamma = gamma_from_impedance(z, z0)
        z_back = impedance_from_gamma(gamma, z0)
        np.testing.assert_allclose(z_back, z, rtol=1e-10, atol=1e-10)

    def test_invalid_denominator_near_negative_z0(self) -> None:
        z0 = 50.0 + 0j
        z = np.array([-50.0 + 1e-15j], dtype=np.complex128)
        gamma = gamma_from_impedance(z, z0, denominator_epsilon=1e-12)
        self.assertTrue(np.isnan(gamma[0].real))
        self.assertTrue(np.isnan(gamma[0].imag))


if __name__ == "__main__":
    unittest.main()
