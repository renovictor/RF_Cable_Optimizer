from __future__ import annotations

import unittest

import numpy as np

from rf.two_port import (
    SParameters,
    evaluate_match_efficiency,
    matcher_output_impedance,
    s_to_z,
    solve_loaded_two_port,
    z_to_abcd,
)


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

    def test_screenshot_regression_z_conversion_and_zout(self) -> None:
        s = SParameters(
            s11=complex(-0.817, 0.414),
            s12=complex(-0.0561, 0.343),
            s21=complex(-0.0559, 0.343),
            s22=complex(-0.670, -0.625),
            zref_ohm=50.0,
        )
        z = s_to_z(s)
        self.assertAlmostEqual(z[0, 0].real, 0.590, places=3)
        self.assertAlmostEqual(z[0, 0].imag, 11.175, places=3)
        self.assertAlmostEqual(z[0, 1].real, -0.216, places=3)
        self.assertAlmostEqual(z[0, 1].imag, 10.096, places=3)
        self.assertAlmostEqual(z[1, 0].real, -0.211, places=3)
        self.assertAlmostEqual(z[1, 0].imag, 10.095, places=3)
        self.assertAlmostEqual(z[1, 1].real, 0.586, places=3)
        self.assertAlmostEqual(z[1, 1].imag, -19.314, places=3)
        zout = matcher_output_impedance(z, 50.0 + 0.0j)
        self.assertAlmostEqual(zout.real, 2.524, places=3)
        self.assertAlmostEqual(zout.imag, -19.657, places=3)

    def test_loaded_two_port_power_sign_convention(self) -> None:
        s = SParameters(
            s11=complex(-0.817, 0.414),
            s12=complex(-0.0561, 0.343),
            s21=complex(-0.0559, 0.343),
            s22=complex(-0.670, -0.625),
            zref_ohm=50.0,
        )
        loaded = solve_loaded_two_port(
            s=s,
            z_source=50.0 + 0.0j,
            z_load=complex(2.62069, 19.702),
            p_match_input_w=1300.0,
        )
        self.assertGreater(loaded.p_match_input_w, 0.0)
        self.assertGreaterEqual(loaded.p_match_output_w, 0.0)
        self.assertAlmostEqual(
            loaded.p_match_input_w,
            loaded.p_match_output_w + loaded.p_match_loss_w,
            places=6,
        )
        self.assertAlmostEqual(
            loaded.p_match_output_w,
            -float(np.real(loaded.v2_rms_v * np.conj(loaded.i2_into_matcher_a))),
            places=6,
        )
        self.assertAlmostEqual(
            loaded.p_match_input_w,
            float(np.real(loaded.v1_rms_v * np.conj(loaded.i1_into_matcher_a))),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
