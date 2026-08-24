import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.meshing.validate_corpus import CorpusError, validate_manifest


class CorpusValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        self.fixture = self.fixtures / "shape.step"
        self.fixture.write_bytes(b"STEP fixture")
        self.test_source = self.root / "tests.rs"
        self.test_source.write_text("fn corpus_shape_is_valid() {}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def document(self):
        return {
            "schema_version": 1,
            "corpus_revision": 1,
            "fixture_root": "fixtures",
            "entries": [
                {
                    "id": "shape",
                    "path": "shape.step",
                    "sha256": hashlib.sha256(b"STEP fixture").hexdigest(),
                    "tier": "small",
                    "format": "step",
                    "supported_input": True,
                    "provenance": {
                        "origin": "test fixture",
                        "exporter": "test exporter",
                        "exporter_version": "1",
                        "license": "test license",
                    },
                    "features": ["analytic-primitives"],
                    "expected": {
                        "outcome": "success",
                        "topology": "one solid",
                        "mass_properties": "unit volume",
                        "regions": 1,
                        "mesh_error_bounds": "request bounds",
                    },
                    "test": {"source": "tests.rs", "name": "corpus_shape_is_valid"},
                }
            ],
        }

    def write(self, document):
        manifest = self.root / "corpus.json"
        manifest.write_text(json.dumps(document))
        return manifest

    def test_accepts_complete_immutable_entry(self):
        self.assertEqual(validate_manifest(self.write(self.document()), self.root), 1)

    def test_rejects_fixture_digest_drift(self):
        document = self.document()
        document["entries"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CorpusError, "SHA-256"):
            validate_manifest(self.write(document), self.root)

    def test_rejects_unregistered_fixture(self):
        (self.fixtures / "unregistered.brep").write_bytes(b"BREP")
        with self.assertRaisesRegex(CorpusError, "unregistered"):
            validate_manifest(self.write(self.document()), self.root)

    def test_rejects_expected_failure_for_supported_input(self):
        document = self.document()
        document["entries"][0]["expected"]["outcome"] = "admission-rejection"
        with self.assertRaisesRegex(CorpusError, "supported inputs"):
            validate_manifest(self.write(document), self.root)

    def test_rejects_missing_executable_test_anchor(self):
        document = self.document()
        document["entries"][0]["test"]["name"] = "not_a_test"
        with self.assertRaisesRegex(CorpusError, "test anchor"):
            validate_manifest(self.write(document), self.root)


if __name__ == "__main__":
    unittest.main()
