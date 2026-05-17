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
                "em_homogeneous_sigma_omega_response_coverage_ratio",
            ),
            (
                "electromagnetic_reference_homogeneous_gpu_provider",
                "em_homogeneous_flux_divergence_proxy",
            ),
            (
                "electromagnetic_reference_homogeneous_gpu_provider",
                "em_homogeneous_boundary_energy_ratio",
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_sigma_omega_scale_spread_ratio",
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_sigma_omega_response_coverage_ratio",
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_region_contrast_index",
            ),
            (
                "electromagnetic_reference_heterogeneous_gpu_provider",
                "em_heterogeneous_source_realization_ratio",
            ),
            (
                "electromagnetic_reference_boundary_penalty_stress_gpu_provider",
                "em_boundary_penalty_conditioning_contribution",
            ),
            (
                "electromagnetic_reference_multi_region_phased_source_gpu_provider",
                "em_phased_source_energy_consistency_ratio",
            ),
            ("cfd_steady_gpu_provider", "cfd_reference_density_kg_per_m3"),
            ("cfd_steady_gpu_provider", "cfd_reynolds_proxy"),
            ("cht_coupled_gpu_provider", "cht_applied_temperature_delta_k"),
            ("cht_coupled_gpu_provider", "cht_reynolds_proxy"),
            ("fsi_coupled_gpu_provider", "fsi_reynolds_proxy"),
            ("fsi_coupled_gpu_provider", "fsi_structural_step_count"),
            (
                "thermo_gradient_pathological_gpu_provider",
                "thermo_gradient_pathological_spread_ratio",
            ),
            (
                "thermo_shock_oscillatory_gpu_provider",
                "thermo_shock_oscillatory_temporal_variation",
            ),
            (
                "electro_thermal_joule_pathological_gpu_provider",
                "electro_thermal_pathological_conductivity_spread_ratio",
            ),
            (
                "nonlinear_plastic_hardening_reference_complex_gpu_provider",
                "plasticity_hardening_reference_complex_load_realization_ratio",
            ),
            (
                "nonlinear_contact_frictionless_reference_complex_gpu_provider",
                "contact_frictionless_complex_load_amplification_ratio",
            ),
            ("acoustic_harmonic_gpu_provider", "acoustic_mode_count"),
            ("acoustic_harmonic_gpu_provider", "acoustic_residual_warn_threshold"),
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
