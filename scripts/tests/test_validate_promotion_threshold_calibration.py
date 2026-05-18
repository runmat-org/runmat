import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.analysis.governance.validate_promotion_threshold_calibration import main


def _payload(generated_at: str, source_report_count: int = 4):
    return {
        "schema_version": "promotion-threshold-calibration/v1",
        "generated_at": generated_at,
        "rationale": "rolling_median_reference_fixtures",
        "source_report_count": source_report_count,
        "source_trusted_report_count": source_report_count,
        "cadence_days": {"release": 7, "development": 14, "feature": 30},
        "max_missed_cycles_allowed": 1,
        "by_profile": {
            "release": {
                "plastic_promotion_max_blockers": 0,
                "contact_promotion_max_blockers": 0,
                "promotion_max_blocker_regression": 0,
                "rolling_report_count": source_report_count,
                "rolling_trusted_report_count": source_report_count,
            },
            "development": {
                "plastic_promotion_max_blockers": 1,
                "contact_promotion_max_blockers": 1,
                "promotion_max_blocker_regression": 0,
                "rolling_report_count": source_report_count,
                "rolling_trusted_report_count": source_report_count,
            },
            "feature": {
                "plastic_promotion_max_blockers": 2,
                "contact_promotion_max_blockers": 2,
                "promotion_max_blocker_regression": 1,
                "rolling_report_count": source_report_count,
                "rolling_trusted_report_count": source_report_count,
            },
        },
    }


class ValidatePromotionThresholdCalibrationTests(unittest.TestCase):
    def test_passes_for_fresh_valid_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion_calibration.json"
            payload = _payload(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            path.write_text(json.dumps(payload))

            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT"] = str(path)
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE"] = "true"
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_REQUIRE_CADENCE"] = "true"
            os.environ["GITHUB_REF_NAME"] = "main"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_REQUIRE_CADENCE", None)
                os.environ.pop("GITHUB_REF_NAME", None)
            self.assertEqual(rc, 0)

    def test_fails_when_stale_under_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion_calibration.json"
            stale = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat().replace(
                "+00:00", "Z"
            )
            path.write_text(json.dumps(_payload(stale)))

            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT"] = str(path)
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE"] = "true"
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MAX_AGE_DAYS"] = "7"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MAX_AGE_DAYS", None)
            self.assertEqual(rc, 1)

    def test_fails_when_trusted_counts_exceed_raw_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion_calibration.json"
            payload = _payload(
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                source_report_count=2,
            )
            payload["source_trusted_report_count"] = 3
            payload["by_profile"]["release"]["rolling_trusted_report_count"] = 3
            path.write_text(json.dumps(payload))

            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT"] = str(path)
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", None)
            self.assertEqual(rc, 1)

    def test_fails_when_trusted_source_count_below_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion_calibration.json"
            payload = _payload(
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                source_report_count=4,
            )
            payload["source_trusted_report_count"] = 1
            for entry in payload["by_profile"].values():
                entry["rolling_report_count"] = 4
                entry["rolling_trusted_report_count"] = 1
            path.write_text(json.dumps(payload))

            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT"] = str(path)
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE"] = "true"
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MIN_SOURCE_REPORTS"] = "0"
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MIN_TRUSTED_SOURCE_REPORTS"] = "2"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MIN_SOURCE_REPORTS", None)
                os.environ.pop(
                    "RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MIN_TRUSTED_SOURCE_REPORTS",
                    None,
                )
            self.assertEqual(rc, 1)

    def test_fails_when_rationale_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion_calibration.json"
            payload = _payload(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            payload["rationale"] = "custom_rationale"
            path.write_text(json.dumps(payload))

            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT"] = str(path)
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", None)
            self.assertEqual(rc, 1)

    def test_fails_when_profile_sets_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion_calibration.json"
            payload = _payload(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            payload["by_profile"].pop("feature")
            payload["cadence_days"].pop("feature")
            path.write_text(json.dumps(payload))

            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT"] = str(path)
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", None)
            self.assertEqual(rc, 1)

    def test_fails_when_profile_sets_have_unexpected_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion_calibration.json"
            payload = _payload(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            payload["by_profile"]["experimental"] = payload["by_profile"]["feature"].copy()
            payload["cadence_days"]["experimental"] = 90
            path.write_text(json.dumps(payload))

            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT"] = str(path)
            os.environ["RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT", None)
                os.environ.pop("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", None)
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
