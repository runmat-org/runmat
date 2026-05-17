import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.governance.validate_external_reference_benchmark import main


class ValidateExternalReferenceBenchmarkTests(unittest.TestCase):
    def test_passes_with_valid_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external_reference_benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "external-reference-benchmark/v1",
                        "scenario_id": "m6_elastoplastic_contact_bracket_v1",
                        "reference_source": {
                            "primary": "nafems_elastoplastic_case",
                            "secondary": "calculix_cross_check",
                        },
                        "generated_at": "2026-03-10T00:00:00Z",
                        "metrics": [
                            {
                                "name": "cfd_reference_density_kg_per_m3",
                                "fixture_id": "cfd_steady_gpu_provider",
                                "observed": 1.0,
                                "reference": 1.02,
                                "pass": True,
                            },
                            {
                                "name": "cfd_reynolds_proxy",
                                "fixture_id": "cfd_steady_gpu_provider",
                                "observed": 340000.0,
                                "reference": 338000.0,
                                "pass": True,
                            },
                            {
                                "name": "cht_applied_temperature_delta_k",
                                "fixture_id": "cht_coupled_gpu_provider",
                                "observed": 60.0,
                                "reference": 60.0,
                                "pass": True,
                            },
                            {
                                "name": "cht_reynolds_proxy",
                                "fixture_id": "cht_coupled_gpu_provider",
                                "observed": 338000.0,
                                "reference": 338000.0,
                                "pass": True,
                            },
                            {
                                "name": "fsi_reynolds_proxy",
                                "fixture_id": "fsi_coupled_gpu_provider",
                                "observed": 271000.0,
                                "reference": 270000.0,
                                "pass": True,
                            },
                            {
                                "name": "fsi_structural_step_count",
                                "fixture_id": "fsi_coupled_gpu_provider",
                                "observed": 1.0,
                                "reference": 1.0,
                                "pass": True,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH"] = str(path)
            os.environ["RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH", None)
                os.environ.pop("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE", None)
            self.assertEqual(rc, 0)

    def test_fails_in_enforce_mode_when_missing(self):
        os.environ[
            "RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH"
        ] = "/tmp/does-not-exist-benchmark.json"
        os.environ["RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE"] = "true"
        try:
            rc = main()
        finally:
            os.environ.pop("RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH", None)
            os.environ.pop("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE", None)
        self.assertEqual(rc, 1)

    def test_fails_when_require_pass_and_metric_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external_reference_benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "external-reference-benchmark/v1",
                        "scenario_id": "m6_elastoplastic_contact_bracket_v1",
                        "reference_source": {
                            "primary": "nafems_elastoplastic_case",
                            "secondary": "calculix_cross_check",
                        },
                        "generated_at": "2026-03-10T00:00:00Z",
                        "metrics": [
                            {
                                "name": "cfd_reference_density_kg_per_m3",
                                "fixture_id": "cfd_steady_gpu_provider",
                                "observed": 1.0,
                                "reference": 1.02,
                                "pass": False,
                            },
                            {
                                "name": "cfd_reynolds_proxy",
                                "fixture_id": "cfd_steady_gpu_provider",
                                "observed": 340000.0,
                                "reference": 338000.0,
                                "pass": True,
                            },
                            {
                                "name": "cht_applied_temperature_delta_k",
                                "fixture_id": "cht_coupled_gpu_provider",
                                "observed": 60.0,
                                "reference": 60.0,
                                "pass": True,
                            },
                            {
                                "name": "cht_reynolds_proxy",
                                "fixture_id": "cht_coupled_gpu_provider",
                                "observed": 338000.0,
                                "reference": 338000.0,
                                "pass": True,
                            },
                            {
                                "name": "fsi_reynolds_proxy",
                                "fixture_id": "fsi_coupled_gpu_provider",
                                "observed": 271000.0,
                                "reference": 270000.0,
                                "pass": True,
                            },
                            {
                                "name": "fsi_structural_step_count",
                                "fixture_id": "fsi_coupled_gpu_provider",
                                "observed": 1.0,
                                "reference": 1.0,
                                "pass": True,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH"] = str(path)
            os.environ["RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE"] = "true"
            os.environ["RUNMAT_VALIDATE_EXTERNAL_REFERENCE_REQUIRE_PASS"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH", None)
                os.environ.pop("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE", None)
                os.environ.pop("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_REQUIRE_PASS", None)
            self.assertEqual(rc, 1)

    def test_fails_when_required_fixture_metrics_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external_reference_benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "external-reference-benchmark/v1",
                        "scenario_id": "m6_elastoplastic_contact_bracket_v1",
                        "reference_source": {
                            "primary": "nafems_elastoplastic_case",
                            "secondary": "calculix_cross_check",
                        },
                        "generated_at": "2026-03-10T00:00:00Z",
                        "metrics": [
                            {
                                "name": "cfd_reference_density_kg_per_m3",
                                "fixture_id": "cfd_steady_gpu_provider",
                                "observed": 1.0,
                                "reference": 1.02,
                                "pass": True,
                            }
                        ],
                    }
                )
            )
            os.environ["RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH"] = str(path)
            os.environ["RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE"] = "true"
            try:
                rc = main()
            finally:
                os.environ.pop("RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH", None)
                os.environ.pop("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE", None)
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
