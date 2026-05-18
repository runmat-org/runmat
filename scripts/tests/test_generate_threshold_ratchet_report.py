import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.governance.generate_threshold_ratchet_report import main


class GenerateThresholdRatchetReportTests(unittest.TestCase):
    def test_generates_report_from_readiness_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "readiness.json"
            out_json = root / "ratchet.json"
            out_md = root / "ratchet.md"
            inp.write_text(
                json.dumps(
                    {
                        "governance_profile": "development",
                        "reference_trend_rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 6,
                        "rolling_trusted_report_count": 4,
                        "plastic_trend_ratio": 1.02,
                        "plastic_reference_trend_ratio": 1.03,
                        "contact_trend_ratio": 1.01,
                        "contact_reference_trend_ratio": 1.04,
                        "thermal_spread_trend_ratio": 1.02,
                    }
                )
            )

            import os

            os.environ["RUNMAT_THRESHOLD_RATCHET_INPUT"] = str(inp)
            os.environ["RUNMAT_THRESHOLD_RATCHET_OUTPUT_JSON"] = str(out_json)
            os.environ["RUNMAT_THRESHOLD_RATCHET_OUTPUT_MD"] = str(out_md)
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_INPUT", None)
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_OUTPUT_JSON", None)
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_OUTPUT_MD", None)

            self.assertEqual(rc, 0)
            report = json.loads(out_json.read_text())
            self.assertEqual(report["governance_profile"], "development")
            self.assertEqual(report["schema_version"], "threshold-ratchet-report/v1")
            self.assertEqual(report["rolling_report_count"], 6)
            self.assertEqual(report["rolling_trusted_report_count"], 4)
            self.assertEqual(len(report["entries"]), 5)
            self.assertTrue(
                any(
                    entry["threshold_key"]
                    == "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_TREND_RATIO"
                    for entry in report["entries"]
                )
            )
            markdown = out_md.read_text()
            self.assertIn("Threshold Ratchet Report", markdown)
            self.assertIn("Rolling reports (raw/trusted): `6`/`4`", markdown)

    def test_falls_back_to_raw_count_when_trusted_count_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "readiness.json"
            out_json = root / "ratchet.json"
            out_md = root / "ratchet.md"
            inp.write_text(
                json.dumps(
                    {
                        "governance_profile": "release",
                        "reference_trend_rationale": "rolling_median_reference_fixtures",
                        "rolling_report_count": 3,
                        "plastic_trend_ratio": 1.0,
                        "plastic_reference_trend_ratio": 1.0,
                        "contact_trend_ratio": 1.0,
                        "contact_reference_trend_ratio": 1.0,
                        "thermal_spread_trend_ratio": 1.0,
                    }
                )
            )

            import os

            os.environ["RUNMAT_THRESHOLD_RATCHET_INPUT"] = str(inp)
            os.environ["RUNMAT_THRESHOLD_RATCHET_OUTPUT_JSON"] = str(out_json)
            os.environ["RUNMAT_THRESHOLD_RATCHET_OUTPUT_MD"] = str(out_md)
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_INPUT", None)
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_OUTPUT_JSON", None)
                os.environ.pop("RUNMAT_THRESHOLD_RATCHET_OUTPUT_MD", None)

            self.assertEqual(rc, 0)
            report = json.loads(out_json.read_text())
            self.assertEqual(report["rolling_report_count"], 3)
            self.assertEqual(report["rolling_trusted_report_count"], 3)
            self.assertIn("Rolling reports (raw/trusted): `3`/`3`", out_md.read_text())


if __name__ == "__main__":
    unittest.main()
