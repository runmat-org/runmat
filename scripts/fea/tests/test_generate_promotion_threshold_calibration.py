import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.fea.governance.generate_promotion_threshold_calibration import main
from scripts.fea.governance.validate_promotion_threshold_calibration import (
    EXPECTED_PROFILES,
    EXPECTED_RATIONALE,
)


class GeneratePromotionThresholdCalibrationTests(unittest.TestCase):
    def test_generates_calibration_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rolling = root / "rolling"
            rolling.mkdir(parents=True, exist_ok=True)
            sample = {
                "records": [
                    {
                        "fixture_id": "nonlinear_plastic_hardening_reference_gpu_provider",
                        "plastic_nonlinear_severity": 0.2,
                    },
                    {
                        "fixture_id": "nonlinear_contact_frictionless_reference_gpu_provider",
                        "contact_nonlinear_severity": 0.2,
                    },
                ]
            }
            (rolling / "analysis_benchmark_report_rolling_1.json").write_text(
                json.dumps(sample)
            )
            out = root / "calibration.json"

            os.environ["RUNMAT_ANALYSIS_BASELINE_DIR"] = str(rolling)
            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_OUTPUT"] = str(out)
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_ANALYSIS_BASELINE_DIR", None)
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_OUTPUT", None)

            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["schema_version"], "promotion-threshold-calibration/v1")
            self.assertIn("by_profile", payload)
            self.assertEqual(payload["source_report_count"], 1)
            self.assertEqual(payload["source_trusted_report_count"], 1)
            self.assertEqual(
                payload["by_profile"]["release"]["rolling_trusted_report_count"], 1
            )
            self.assertEqual(payload["rationale"], EXPECTED_RATIONALE)
            self.assertEqual(set(payload["by_profile"].keys()), set(EXPECTED_PROFILES))
            self.assertEqual(set(payload["cadence_days"].keys()), set(EXPECTED_PROFILES))
            self.assertGreaterEqual(payload["max_missed_cycles_allowed"], 0)
            for profile in EXPECTED_PROFILES:
                self.assertEqual(
                    payload["by_profile"][profile]["rolling_report_count"],
                    payload["source_report_count"],
                )
                self.assertEqual(
                    payload["by_profile"][profile]["rolling_trusted_report_count"],
                    payload["source_trusted_report_count"],
                )

    def test_filters_untrusted_rolling_reports_and_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rolling = root / "rolling"
            rolling.mkdir(parents=True, exist_ok=True)

            trusted_low = {
                "passed": True,
                "publishable": True,
                "records": [
                    {
                        "fixture_id": "nonlinear_plastic_hardening_reference_gpu_provider",
                        "plastic_nonlinear_severity": 0.2,
                    },
                    {
                        "fixture_id": "nonlinear_plastic_hardening_reference_complex_gpu_provider",
                        "plastic_nonlinear_severity": 0.2,
                    },
                    {
                        "fixture_id": "nonlinear_contact_frictionless_reference_gpu_provider",
                        "contact_nonlinear_severity": 0.2,
                    },
                    {
                        "fixture_id": "nonlinear_contact_frictionless_reference_complex_gpu_provider",
                        "contact_nonlinear_severity": 0.2,
                    },
                ],
            }
            trusted_low_with_unpublishable_record = {
                "passed": True,
                "publishable": True,
                "records": [
                    {
                        "fixture_id": "nonlinear_plastic_hardening_reference_gpu_provider",
                        "plastic_nonlinear_severity": 0.2,
                    },
                    {
                        "fixture_id": "nonlinear_plastic_hardening_reference_complex_gpu_provider",
                        "plastic_nonlinear_severity": 0.95,
                        "publishable": False,
                    },
                    {
                        "fixture_id": "nonlinear_contact_frictionless_reference_gpu_provider",
                        "contact_nonlinear_severity": 0.2,
                    },
                    {
                        "fixture_id": "nonlinear_contact_frictionless_reference_complex_gpu_provider",
                        "contact_nonlinear_severity": 0.95,
                        "publishable": False,
                    },
                ],
            }
            failed_high = {
                "passed": False,
                "publishable": True,
                "records": [
                    {
                        "fixture_id": "nonlinear_plastic_hardening_reference_gpu_provider",
                        "plastic_nonlinear_severity": 0.95,
                    },
                    {
                        "fixture_id": "nonlinear_contact_frictionless_reference_gpu_provider",
                        "contact_nonlinear_severity": 0.95,
                    },
                ],
            }
            nonpublishable_high = {
                "passed": True,
                "publishable": False,
                "records": [
                    {
                        "fixture_id": "nonlinear_plastic_hardening_reference_gpu_provider",
                        "plastic_nonlinear_severity": 0.95,
                    },
                    {
                        "fixture_id": "nonlinear_contact_frictionless_reference_gpu_provider",
                        "contact_nonlinear_severity": 0.95,
                    },
                ],
            }

            (rolling / "analysis_benchmark_report_rolling_1.json").write_text(
                json.dumps(trusted_low)
            )
            (rolling / "analysis_benchmark_report_rolling_2.json").write_text(
                json.dumps(trusted_low_with_unpublishable_record)
            )
            (rolling / "analysis_benchmark_report_rolling_3.json").write_text(
                json.dumps(failed_high)
            )
            (rolling / "analysis_benchmark_report_rolling_4.json").write_text(
                json.dumps(nonpublishable_high)
            )
            out = root / "calibration.json"

            os.environ["RUNMAT_ANALYSIS_BASELINE_DIR"] = str(rolling)
            os.environ["RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_OUTPUT"] = str(out)
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_ANALYSIS_BASELINE_DIR", None)
                os.environ.pop("RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_OUTPUT", None)

            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["source_report_count"], 4)
            self.assertEqual(payload["source_trusted_report_count"], 2)

            feature = payload["by_profile"]["feature"]
            self.assertEqual(feature["rolling_report_count"], 4)
            self.assertEqual(feature["rolling_trusted_report_count"], 2)
            self.assertEqual(feature["plastic_promotion_max_blockers"], 1)
            self.assertEqual(feature["contact_promotion_max_blockers"], 1)
            self.assertLessEqual(
                payload["by_profile"]["release"]["plastic_promotion_max_blockers"],
                payload["by_profile"]["development"]["plastic_promotion_max_blockers"],
            )
            self.assertLessEqual(
                payload["by_profile"]["development"]["plastic_promotion_max_blockers"],
                payload["by_profile"]["feature"]["plastic_promotion_max_blockers"],
            )
            self.assertLessEqual(
                payload["by_profile"]["release"]["contact_promotion_max_blockers"],
                payload["by_profile"]["development"]["contact_promotion_max_blockers"],
            )
            self.assertLessEqual(
                payload["by_profile"]["development"]["contact_promotion_max_blockers"],
                payload["by_profile"]["feature"]["contact_promotion_max_blockers"],
            )
            self.assertLessEqual(
                payload["by_profile"]["release"]["promotion_max_blocker_regression"],
                payload["by_profile"]["development"]["promotion_max_blocker_regression"],
            )
            self.assertLessEqual(
                payload["by_profile"]["development"]["promotion_max_blocker_regression"],
                payload["by_profile"]["feature"]["promotion_max_blocker_regression"],
            )


if __name__ == "__main__":
    unittest.main()
