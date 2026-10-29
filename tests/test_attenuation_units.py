from __future__ import annotations

import unittest

from models.cable_parameters import AttenuationUnit
from rf.transmission_line import db_per_m_to_np_per_m, to_db_per_m


class AttenuationUnitTests(unittest.TestCase):
    def test_db_unit_conversion(self) -> None:
        self.assertAlmostEqual(to_db_per_m(1.0, AttenuationUnit.DB_PER_M), 1.0, places=12)
        self.assertAlmostEqual(to_db_per_m(100.0, AttenuationUnit.DB_PER_100M), 1.0, places=12)
        self.assertAlmostEqual(to_db_per_m(30.48, AttenuationUnit.DB_PER_100FT), 1.0, places=12)

    def test_np_conversion(self) -> None:
        self.assertAlmostEqual(db_per_m_to_np_per_m(8.685889638), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()

