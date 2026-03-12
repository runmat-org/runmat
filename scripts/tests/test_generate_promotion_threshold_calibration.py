import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.governance.generate_promotion_threshold_calibration import main


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


if __name__ == "__main__":
    unittest.main()
