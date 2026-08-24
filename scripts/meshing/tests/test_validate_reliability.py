import json
import tempfile
import unittest
from pathlib import Path

from scripts.meshing.validate_reliability import ReliabilityError, validate_catalog


class ReliabilityValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "test.rs").write_text("fn reliability_passes() {}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def document(self):
        return {
            "schema_version": 1,
            "catalog_revision": 1,
            "required_controls": ["cancellation"],
            "cases": [
                {
                    "id": "case",
                    "owner": "meshing-execution",
                    "controls": ["cancellation"],
                    "expected": "typed cancellation",
                    "test": {"source": "test.rs", "name": "reliability_passes"},
                }
            ],
        }

    def write(self, document):
        path = self.root / "reliability.json"
        path.write_text(json.dumps(document))
        return path

    def test_accepts_complete_owned_control_inventory(self):
        self.assertEqual(validate_catalog(self.write(self.document()), self.root), (1, 1))

    def test_rejects_missing_required_control(self):
        document = self.document()
        document["required_controls"].append("missing")
        with self.assertRaisesRegex(ReliabilityError, "lack executable coverage"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_unknown_control_typo(self):
        document = self.document()
        document["cases"][0]["controls"] = ["typo"]
        with self.assertRaisesRegex(ReliabilityError, "unknown controls"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_invalid_domain_owner(self):
        document = self.document()
        document["cases"][0]["owner"] = "meshing-scheduler"
        with self.assertRaisesRegex(ReliabilityError, "invalid domain owner"):
            validate_catalog(self.write(document), self.root)


if __name__ == "__main__":
    unittest.main()
