from __future__ import annotations

import math
import unittest

from rf.coax import inner_diameter_for_target_z0, z0_from_geometry


class CoaxTests(unittest.TestCase):
    def test_geometry_formula(self) -> None:
        d = 5.0
        D = 20.0
        er = 2.1
        z0 = z0_from_geometry(D, d, er)
        expected = (60.0 / math.sqrt(er)) * math.log(D / d)
        self.assertAlmostEqual(z0, expected, places=12)

    def test_target_inner_diameter(self) -> None:
        target = 30.0
        D = 20.0
        er = 2.1
        d = inner_diameter_for_target_z0(target, D, er)
        z0 = z0_from_geometry(D, d, er)
        self.assertAlmostEqual(z0, target, places=10)


if __name__ == "__main__":
    unittest.main()

