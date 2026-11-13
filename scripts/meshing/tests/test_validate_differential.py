import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.meshing.validate_differential import DifferentialError, validate_catalog


class DifferentialValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "fixture.brep").write_bytes(b"fixture")
        (self.root / "test.rs").write_text("fn comparison_passes() {}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def document(self):
        return {
            "schema_version": 1,
            "catalog_revision": 1,
            "revision_explanation": "initial measured baseline",
            "required_comparisons": ["topology"],
            "cases": [
                {
                    "id": "case",
                    "fixture": "fixture.brep",
                    "fixture_sha256": hashlib.sha256(b"fixture").hexdigest(),
                    "reference_meshers": [
                        {
                            "name": "independent",
                            "implementation": "separate implementation",
                            "trusted": True,
                            "independent_of_runmat_generator": True,
                        }
                    ],
                    "comparisons": {"topology": "same closed topology"},
                    "mismatches": [
                        {
                            "metric": "triangle-count",
                            "runmat": "16",
                            "reference": "12",
                            "disposition": "accepted",
                            "explanation": "different legal diagonals",
                        }
                    ],
                    "test": {"source": "test.rs", "name": "comparison_passes"},
                }
            ],
        }

    def write(self, document):
        path = self.root / "differential.json"
        path.write_text(json.dumps(document))
        return path

    def test_accepts_independent_comparison_with_disposition(self):
        self.assertEqual(validate_catalog(self.write(self.document()), self.root), 1)

    def test_rejects_reference_using_runmat_generator(self):
        document = self.document()
        document["cases"][0]["reference_meshers"][0]["independent_of_runmat_generator"] = False
        with self.assertRaisesRegex(DifferentialError, "trusted and independent"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_missing_required_comparison(self):
        document = self.document()
        document["cases"][0]["comparisons"] = {}
        with self.assertRaisesRegex(DifferentialError, "comparison inventory"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_undispositioned_mismatch(self):
        document = self.document()
        document["cases"][0]["mismatches"][0]["disposition"] = "pending"
        with self.assertRaisesRegex(DifferentialError, "reviewed disposition"):
            validate_catalog(self.write(document), self.root)

    def test_rejects_baseline_revision_without_explanation(self):
        document = self.document()
        document["revision_explanation"] = ""
        with self.assertRaisesRegex(DifferentialError, "revision_explanation"):
            validate_catalog(self.write(document), self.root)


if __name__ == "__main__":
    unittest.main()
