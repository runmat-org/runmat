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
            "    timeout-minutes: 10",
            "    continue-on-error: true\n    timeout-minutes: 10",
        )
        with self.assertRaisesRegex(ReleaseEvidenceError, "contains bypass"):
            validate_workflow(path)

    def test_rejects_scoping_required_receipts_to_the_deferred_meshing_cohort(self):
        path = self.mutated_workflow(
            "run: python3 scripts/qualification/scientific.py verify\n",
            "run: python3 scripts/qualification/scientific.py verify --cohort meshing-release\n",
        )
        with self.assertRaisesRegex(ReleaseEvidenceError, "all required scientific receipts"):
            validate_workflow(path)

    def test_rejects_missing_attestation(self):
        path = self.mutated_workflow("        uses: actions/attest@v4", "        uses: actions/checkout@v4")
        with self.assertRaisesRegex(ReleaseEvidenceError, "actions/attest"):
            validate_workflow(path)

    def test_rejects_promotion_without_evidence_dependency(self):
        path = self.mutated_workflow(
            "needs: [release-validation, meshing-release-evidence, build-release]",
            "needs: [release-validation, build-release]",
        )
        with self.assertRaisesRegex(ReleaseEvidenceError, "promotion"):
            validate_workflow(path)


if __name__ == "__main__":
    unittest.main()
