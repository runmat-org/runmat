import json
import tempfile
import unittest
from pathlib import Path

from scripts.meshing.validate_conformance import ConformanceError, validate_catalog


class ConformanceValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "tests.rs").write_text("fn difficult_case_passes() {}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def document(self):
        return {
            "schema_version": 1,
            "catalog_revision": 1,
            "required_features": ["difficult-geometry"],
            "cases": [
                {
                    "id": "case",
                    "tier": "small",
                    "features": ["difficult-geometry"],
                    "supported_input": True,
                    "expected": {
                        "outcome": "success",
                        "topology": "closed",
                        "mass_properties": "unit volume",
                        "regions": "one",
                        "error_bounds": "request bounds",
                    },
                    "test": {"source": "tests.rs", "name": "difficult_case_passes"},
                }
            ],
        }

    def write(self, document):
        path = self.root / "conformance.json"
        path.write_text(json.dumps(document))
        return path

    def test_accepts_complete_feature_coverage(self):
        self.assertEqual(validate_catalog(self.write(self.document()), self.root), (1, 1))

    def test_rejects_missing_required_feature_coverage(self):
        document = self.document()
        document["required_features"].append("missing")
        with self.assertRaisesRegex(ConformanceError, "lack executable coverage"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_unknown_feature_typo(self):
        document = self.document()
        document["cases"][0]["features"] = ["typo"]
        with self.assertRaisesRegex(ConformanceError, "unknown features"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_expected_failure_for_supported_input(self):
        document = self.document()
        document["cases"][0]["expected"]["outcome"] = "failure"
        with self.assertRaisesRegex(ConformanceError, "supported inputs"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_missing_test_anchor(self):
        document = self.document()
        document["cases"][0]["test"]["name"] = "missing"
        with self.assertRaisesRegex(ConformanceError, "test anchor"):
            validate_catalog(self.write(document), self.root)


if __name__ == "__main__":
    unittest.main()
