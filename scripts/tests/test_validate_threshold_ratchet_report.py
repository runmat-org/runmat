import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.governance.validate_threshold_ratchet_report import main


class ValidateThresholdRatchetReportTests(unittest.TestCase):
    def test_passes_for_non_release_ratcheted_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratchet.json"
            path.write_text(
                json.dumps(
                    {
                        "governance_profile": "development",
                        "rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 6,
                        "rolling_trusted_report_count": 4,
                        "entries": [
                            {
                                "threshold_key": "A",
                                "status": "ratcheted",
                                "old": 1.2,
                                "new": 1.1,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_THRESHOLD_RATCHET_REPORT"] = str(path)
            os.environ["RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_REPORT", None)
                os.environ.pop("RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE", None)
            self.assertEqual(rc, 0)

    def test_fails_when_non_release_not_ratcheted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratchet.json"
            path.write_text(
                json.dumps(
                    {
                        "governance_profile": "feature",
                        "rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 6,
                        "rolling_trusted_report_count": 4,
                        "entries": [
                            {
                                "threshold_key": "A",
                                "status": "unchanged",
                                "old": 1.2,
                                "new": 1.2,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_THRESHOLD_RATCHET_REPORT"] = str(path)
            os.environ["RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_REPORT", None)
                os.environ.pop("RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE", None)
            self.assertEqual(rc, 1)

    def test_release_allows_unchanged_when_non_regressive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratchet.json"
            path.write_text(
                json.dumps(
                    {
                        "governance_profile": "release",
                        "rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 3,
                        "rolling_trusted_report_count": 3,
                        "entries": [
                            {
                                "threshold_key": "A",
                                "status": "unchanged",
                                "old": 1.1,
                                "new": 1.1,
                                "observed": 1.02,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_THRESHOLD_RATCHET_REPORT"] = str(path)
            os.environ["RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_REPORT", None)
                os.environ.pop("RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE", None)
            self.assertEqual(rc, 0)

    def test_release_fails_when_regressive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratchet.json"
            path.write_text(
                json.dumps(
                    {
                        "governance_profile": "release",
                        "rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 3,
                        "rolling_trusted_report_count": 3,
                        "entries": [
                            {
                                "threshold_key": "A",
                                "status": "unchanged",
                                "old": 1.1,
                                "new": 1.2,
                                "observed": 1.02,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_THRESHOLD_RATCHET_REPORT"] = str(path)
            os.environ["RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_REPORT", None)
                os.environ.pop("RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE", None)
            self.assertEqual(rc, 1)

    def test_require_observed_fails_on_missing_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratchet.json"
            path.write_text(
                json.dumps(
                    {
                        "governance_profile": "release",
                        "rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 3,
                        "rolling_trusted_report_count": 3,
                        "entries": [
                            {
                                "threshold_key": "A",
                                "status": "unchanged",
                                "old": 1.1,
                                "new": 1.1,
                                "observed": None,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_THRESHOLD_RATCHET_REPORT"] = str(path)
            os.environ["RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE"] = "true"
            os.environ["RUNMAT_VALIDATE_THRESHOLD_RATCHET_REQUIRE_OBSERVED"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_REPORT", None)
                os.environ.pop("RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE", None)
                os.environ.pop("RUNMAT_VALIDATE_THRESHOLD_RATCHET_REQUIRE_OBSERVED", None)
            self.assertEqual(rc, 1)

    def test_fails_when_trusted_rolling_count_exceeds_raw_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratchet.json"
            path.write_text(
                json.dumps(
                    {
                        "governance_profile": "release",
                        "rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 2,
                        "rolling_trusted_report_count": 3,
                        "entries": [
                            {
                                "threshold_key": "A",
                                "status": "unchanged",
                                "old": 1.1,
                                "new": 1.1,
                                "observed": 1.02,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_THRESHOLD_RATCHET_REPORT"] = str(path)
            os.environ["RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_REPORT", None)
                os.environ.pop("RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE", None)
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
