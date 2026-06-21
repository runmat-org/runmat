import tempfile
import unittest
from pathlib import Path

from scripts.fea.governance.validate_public_field_ids import main, validate


class ValidatePublicFieldIdsTests(unittest.TestCase):
    def write_contract(self, text: str) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "contracts.rs"
        path.write_text(text, encoding="utf-8")
        return path

    def test_accepts_physical_field_ids(self):
        path = self.write_contract(
            '\n'.join(
                [
                    'pub const FEA_FIELD_CFD_VELOCITY: &str = "cfd.velocity";',
                    'format!("thermal.temperature.{snapshot_index}")',
                    'format!("em.magnetic_flux_density_real")',
                ]
            )
        )

        self.assertEqual(validate([path]), [])
        self.assertEqual(main([str(path)]), 0)

    def test_rejects_proxy_or_placeholder_field_ids(self):
        path = self.write_contract(
            '\n'.join(
                [
                    'pub const BAD_PROXY: &str = "em.vector_potential_proxy";',
                    'pub const BAD_PLACEHOLDER: &str = "thermal.placeholder_temperature";',
                ]
            )
        )

        errors = validate([path])
        self.assertEqual(len(errors), 2)
        self.assertIn("em.vector_potential_proxy", errors[0])
        self.assertIn("thermal.placeholder_temperature", errors[1])
        self.assertEqual(main([str(path)]), 1)

    def test_ignores_non_field_fixture_ids(self):
        path = self.write_contract('"fixture_name_without_field_namespace"')

        self.assertEqual(validate([path]), [])


if __name__ == "__main__":
    unittest.main()
