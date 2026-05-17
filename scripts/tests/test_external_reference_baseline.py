import json
import unittest
from pathlib import Path


class ExternalReferenceBaselineTests(unittest.TestCase):
    def test_coupled_and_em_threshold_metrics_present(self):
        baseline_path = Path("scripts/analysis/reference_data/m6_external_reference_baseline.json")
        payload = json.loads(baseline_path.read_text())
        metrics = payload.get("metrics")
        self.assertIsInstance(metrics, list)

        required = {
            (
                "electromagnetic_reference_homogeneous_gpu_provider",
                "em_homogeneous_sigma_omega_scale_mean",
            ),
            (
                "electromagnetic_reference_homogeneous_gpu_provider",
                "em_homogeneous_flux_divergence_proxy",
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_sigma_omega_scale_spread_ratio",
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_region_contrast_index",
            ),
            ("cfd_steady_gpu_provider", "cfd_reference_density_kg_per_m3"),
            ("cfd_steady_gpu_provider", "cfd_reynolds_proxy"),
            ("cht_coupled_gpu_provider", "cht_applied_temperature_delta_k"),
            ("cht_coupled_gpu_provider", "cht_reynolds_proxy"),
            ("fsi_coupled_gpu_provider", "fsi_reynolds_proxy"),
            ("fsi_coupled_gpu_provider", "fsi_structural_step_count"),
        }
        observed = {
            (metric.get("fixture_id"), metric.get("assertion_name"))
            for metric in metrics
            if metric.get("source") == "threshold_assertion"
        }
        missing = required - observed
        self.assertFalse(
            missing,
            f"missing coupled/em external reference metrics: {sorted(missing)}",
        )


if __name__ == "__main__":
    unittest.main()
