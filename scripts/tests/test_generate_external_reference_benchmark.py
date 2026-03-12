import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.generate_external_reference_benchmark import main


class GenerateExternalReferenceBenchmarkTests(unittest.TestCase):
    def test_generates_benchmark_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "analysis_benchmark_report.json"
            baseline = root / "baseline.json"
            out = root / "external_reference_benchmark.json"

            report.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "fixture_id": "fixture_a",
                                "plastic_nonlinear_severity": 0.25,
                            }
                        ]
                    }
                )
            )
            baseline.write_text(
                json.dumps(
                    {
                        "scenario_id": "m6",
                        "reference_source": {"primary": "p", "secondary": "s"},
                        "metrics": [
                            {
                                "name": "a",
                                "fixture_id": "fixture_a",
                                "field": "plastic_nonlinear_severity",
                                "reference": 0.24,
                                "tolerance_abs": 0.05,
                                "tolerance_rel": 0.5,
                            }
                        ],
                    }
                )
            )

            os.environ["RUNMAT_EXTERNAL_REFERENCE_REPORT_PATH"] = str(report)
            os.environ["RUNMAT_EXTERNAL_REFERENCE_BASELINE_PATH"] = str(baseline)
            os.environ["RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH"] = str(out)
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_EXTERNAL_REFERENCE_REPORT_PATH", None)
                os.environ.pop("RUNMAT_EXTERNAL_REFERENCE_BASELINE_PATH", None)
                os.environ.pop("RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH", None)

            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["schema_version"], "external-reference-benchmark/v1")
            self.assertEqual(len(payload["metrics"]), 1)
            self.assertTrue(payload["metrics"][0]["pass"])


if __name__ == "__main__":
    unittest.main()
