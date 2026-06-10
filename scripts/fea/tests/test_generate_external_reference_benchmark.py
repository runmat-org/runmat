import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.fea.governance.generate_external_reference_benchmark import main


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
                                "threshold_assertions": [
                                    {
                                        "name": "fixture_sigma_omega_scale_mean",
                                        "observed": 1.03,
                                        "passed": True,
                                    }
                                ],
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
                            },
                            {
                                "name": "sigma_omega_scale",
                                "fixture_id": "fixture_a",
                                "field": "unused_for_threshold_source",
                                "source": "threshold_assertion",
                                "assertion_name": "fixture_sigma_omega_scale_mean",
                                "reference": 1.0,
                                "tolerance_abs": 0.05,
                                "tolerance_rel": 0.1,
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
            self.assertEqual(len(payload["metrics"]), 2)
            self.assertTrue(payload["metrics"][0]["pass"])
            self.assertEqual(payload["metrics"][1]["source"], "threshold_assertion")
            self.assertTrue(payload["metrics"][1]["pass"])


if __name__ == "__main__":
    unittest.main()
