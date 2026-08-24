import tempfile
import unittest
from pathlib import Path

from scripts.meshing.validate_release_evidence import (
    DEFAULT_WORKFLOW,
    ReleaseEvidenceError,
    validate_workflow,
)


class ReleaseEvidenceValidationTests(unittest.TestCase):
    def mutated_workflow(self, old: str, new: str) -> Path:
        workflow = DEFAULT_WORKFLOW.read_text().replace(old, new, 1)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "release.yml"
        path.write_text(workflow)
        return path

    def test_repository_release_evidence_is_blocking(self):
        validate_workflow()

    def test_rejects_soft_failure(self):
        path = self.mutated_workflow(
            "    timeout-minutes: 180",
            "    continue-on-error: true\n    timeout-minutes: 180",
        )
        with self.assertRaisesRegex(ReleaseEvidenceError, "contains bypass"):
            validate_workflow(path)

    def test_rejects_missing_attestation(self):
        path = self.mutated_workflow("        uses: actions/attest@v4", "        uses: actions/checkout@v4")
        with self.assertRaisesRegex(ReleaseEvidenceError, "actions/attest"):
            validate_workflow(path)

    def test_rejects_promotion_without_evidence_dependency(self):
        path = self.mutated_workflow(
            "needs: [build-and-test, meshing-release-evidence, build-release]",
            "needs: [build-and-test, build-release]",
        )
        with self.assertRaisesRegex(ReleaseEvidenceError, "promotion"):
            validate_workflow(path)


if __name__ == "__main__":
    unittest.main()
