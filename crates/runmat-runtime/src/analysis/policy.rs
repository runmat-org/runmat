use super::QualityPolicy;

pub(crate) const THERMO_SPREAD_THRESHOLD_BALANCED: f64 = 1.25;
pub(crate) const THERMO_HETEROGENEITY_THRESHOLD_BALANCED: f64 = 0.2;
pub(crate) const EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED: f64 = 2.0;
pub(crate) const EM_HETEROGENEITY_THRESHOLD_BALANCED: f64 = 0.2;
pub(crate) const EM_ASSIGNMENT_COVERAGE_MIN_BALANCED: f64 = 0.85;
pub(crate) const EM_FALLBACK_COEFFICIENT_MAX_BALANCED: f64 = 0.25;
pub(crate) const EM_REGION_CONTRAST_MAX_BALANCED: f64 = 0.85;
pub(crate) const EM_CONDITIONING_MAX_BALANCED: f64 = 2.0e4;
pub(crate) const EM_SOURCE_REALIZATION_MIN_BALANCED: f64 = 0.55;
pub(crate) const EM_SOURCE_REGION_COVERAGE_MIN_BALANCED: f64 = 0.85;
pub(crate) const EM_SOURCE_MATERIAL_ALIGNMENT_MIN_BALANCED: f64 = 0.70;
pub(crate) const EM_SOURCE_OVERLAP_MAX_BALANCED: f64 = 0.75;
pub(crate) const EM_SOURCE_INTERFERENCE_MAX_BALANCED: f64 = 0.55;
pub(crate) const EM_BOUNDARY_ANCHOR_MIN_BALANCED: f64 = 0.45;
pub(crate) const EM_BOUNDARY_LOCALIZATION_MIN_BALANCED: f64 = 0.80;
pub(crate) const EM_GROUND_EFFECTIVENESS_MIN_BALANCED: f64 = 0.70;
pub(crate) const EM_INSULATION_LEAKAGE_MAX_BALANCED: f64 = 0.55;
pub(crate) const EM_FLUX_DIVERGENCE_MAX_BALANCED: f64 = 0.30;
pub(crate) const EM_ENERGY_IMBALANCE_MAX_BALANCED: f64 = 0.40;
pub(crate) const EM_BOUNDARY_ENERGY_MIN_BALANCED: f64 = 0.12;
pub(crate) const EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED: f64 = 0.55;
pub(crate) const EM_SOURCE_REGION_ENERGY_CONSISTENCY_MIN_BALANCED: f64 = 0.65;
pub(crate) const EM_REAL_RESIDUAL_MAX_BALANCED: f64 = 0.30;
pub(crate) const EM_IMAG_RESIDUAL_MAX_BALANCED: f64 = 0.65;
pub(crate) const EM_SWEEP_COUNT_MIN_BALANCED: f64 = 5.0;
pub(crate) const EM_RESONANCE_Q_MIN_BALANCED: f64 = 1.25;

pub(crate) fn thermo_thresholds_for_policy(policy: QualityPolicy) -> (f64, f64) {
    match policy {
        QualityPolicy::Strict => (1.15, 0.12),
        QualityPolicy::Balanced => (
            THERMO_SPREAD_THRESHOLD_BALANCED,
            THERMO_HETEROGENEITY_THRESHOLD_BALANCED,
        ),
        QualityPolicy::Exploratory => (1.4, 0.35),
    }
}

pub(crate) fn thermo_gradient_thresholds_for_policy(policy: QualityPolicy) -> (f64, f64) {
    match policy {
        QualityPolicy::Strict => (0.22, 0.25),
        QualityPolicy::Balanced => (0.30, 0.35),
        QualityPolicy::Exploratory => (0.45, 0.55),
    }
}

pub(crate) fn thermo_field_quality_thresholds_for_policy(policy: QualityPolicy) -> (f64, f64) {
    match policy {
        QualityPolicy::Strict => (0.55, 0.02),
        QualityPolicy::Balanced => (0.45, 0.08),
        QualityPolicy::Exploratory => (0.30, 0.18),
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ElectromagneticQualityThresholds {
    pub(crate) em_spread_threshold: f64,
    pub(crate) em_heterogeneity_threshold: f64,
    pub(crate) em_coverage_min_threshold: f64,
    pub(crate) em_fallback_max_threshold: f64,
    pub(crate) em_contrast_max_threshold: f64,
    pub(crate) em_conditioning_max_threshold: f64,
    pub(crate) em_source_realization_min_threshold: f64,
    pub(crate) em_source_region_coverage_min_threshold: f64,
    pub(crate) em_source_material_alignment_min_threshold: f64,
    pub(crate) em_source_overlap_max_threshold: f64,
    pub(crate) em_source_interference_max_threshold: f64,
    pub(crate) em_boundary_anchor_min_threshold: f64,
    pub(crate) em_boundary_localization_min_threshold: f64,
    pub(crate) em_ground_effectiveness_min_threshold: f64,
    pub(crate) em_insulation_leakage_max_threshold: f64,
    pub(crate) em_divergence_max_threshold: f64,
    pub(crate) em_energy_imbalance_max_threshold: f64,
    pub(crate) em_boundary_energy_min_threshold: f64,
    pub(crate) em_boundary_penalty_contribution_max_threshold: f64,
    pub(crate) em_source_region_energy_consistency_min_threshold: f64,
    pub(crate) em_real_residual_max_threshold: f64,
    pub(crate) em_imag_residual_max_threshold: f64,
}

pub(crate) fn electromagnetic_thresholds_for_policy(
    policy: QualityPolicy,
) -> ElectromagneticQualityThresholds {
    match policy {
        QualityPolicy::Strict => ElectromagneticQualityThresholds {
            em_spread_threshold: 1.5,
            em_heterogeneity_threshold: 0.12,
            em_coverage_min_threshold: 0.95,
            em_fallback_max_threshold: 0.05,
            em_contrast_max_threshold: 0.45,
            em_conditioning_max_threshold: 8.0e3,
            em_source_realization_min_threshold: 0.85,
            em_source_region_coverage_min_threshold: 0.95,
            em_source_material_alignment_min_threshold: 0.9,
            em_source_overlap_max_threshold: 0.55,
            em_source_interference_max_threshold: 0.35,
            em_boundary_anchor_min_threshold: 0.7,
            em_boundary_localization_min_threshold: 0.9,
            em_ground_effectiveness_min_threshold: 0.85,
            em_insulation_leakage_max_threshold: 0.4,
            em_divergence_max_threshold: 0.18,
            em_energy_imbalance_max_threshold: 0.25,
            em_boundary_energy_min_threshold: 0.25,
            em_boundary_penalty_contribution_max_threshold: 0.35,
            em_source_region_energy_consistency_min_threshold: 0.80,
            em_real_residual_max_threshold: 0.18,
            em_imag_residual_max_threshold: 0.50,
        },
        QualityPolicy::Balanced => ElectromagneticQualityThresholds {
            em_spread_threshold: EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED,
            em_heterogeneity_threshold: EM_HETEROGENEITY_THRESHOLD_BALANCED,
            em_coverage_min_threshold: EM_ASSIGNMENT_COVERAGE_MIN_BALANCED,
            em_fallback_max_threshold: EM_FALLBACK_COEFFICIENT_MAX_BALANCED,
            em_contrast_max_threshold: EM_REGION_CONTRAST_MAX_BALANCED,
            em_conditioning_max_threshold: EM_CONDITIONING_MAX_BALANCED,
            em_source_realization_min_threshold: EM_SOURCE_REALIZATION_MIN_BALANCED,
            em_source_region_coverage_min_threshold: EM_SOURCE_REGION_COVERAGE_MIN_BALANCED,
            em_source_material_alignment_min_threshold: EM_SOURCE_MATERIAL_ALIGNMENT_MIN_BALANCED,
            em_source_overlap_max_threshold: EM_SOURCE_OVERLAP_MAX_BALANCED,
            em_source_interference_max_threshold: EM_SOURCE_INTERFERENCE_MAX_BALANCED,
            em_boundary_anchor_min_threshold: EM_BOUNDARY_ANCHOR_MIN_BALANCED,
            em_boundary_localization_min_threshold: EM_BOUNDARY_LOCALIZATION_MIN_BALANCED,
            em_ground_effectiveness_min_threshold: EM_GROUND_EFFECTIVENESS_MIN_BALANCED,
            em_insulation_leakage_max_threshold: EM_INSULATION_LEAKAGE_MAX_BALANCED,
            em_divergence_max_threshold: EM_FLUX_DIVERGENCE_MAX_BALANCED,
            em_energy_imbalance_max_threshold: EM_ENERGY_IMBALANCE_MAX_BALANCED,
            em_boundary_energy_min_threshold: EM_BOUNDARY_ENERGY_MIN_BALANCED,
            em_boundary_penalty_contribution_max_threshold:
                EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED,
            em_source_region_energy_consistency_min_threshold:
                EM_SOURCE_REGION_ENERGY_CONSISTENCY_MIN_BALANCED,
            em_real_residual_max_threshold: EM_REAL_RESIDUAL_MAX_BALANCED,
            em_imag_residual_max_threshold: EM_IMAG_RESIDUAL_MAX_BALANCED,
        },
        QualityPolicy::Exploratory => ElectromagneticQualityThresholds {
            em_spread_threshold: 3.0,
            em_heterogeneity_threshold: 0.35,
            em_coverage_min_threshold: 0.5,
            em_fallback_max_threshold: 0.65,
            em_contrast_max_threshold: 1.8,
            em_conditioning_max_threshold: 1.5e5,
            em_source_realization_min_threshold: 0.2,
            em_source_region_coverage_min_threshold: 0.35,
            em_source_material_alignment_min_threshold: 0.25,
            em_source_overlap_max_threshold: 0.95,
            em_source_interference_max_threshold: 0.85,
            em_boundary_anchor_min_threshold: 0.15,
            em_boundary_localization_min_threshold: 0.4,
            em_ground_effectiveness_min_threshold: 0.35,
            em_insulation_leakage_max_threshold: 0.9,
            em_divergence_max_threshold: 0.8,
            em_energy_imbalance_max_threshold: 1.2,
            em_boundary_energy_min_threshold: 0.03,
            em_boundary_penalty_contribution_max_threshold: 0.9,
            em_source_region_energy_consistency_min_threshold: 0.25,
            em_real_residual_max_threshold: 0.85,
            em_imag_residual_max_threshold: 1.2,
        },
    }
}

pub(crate) fn electromagnetic_sweep_thresholds_for_policy(policy: QualityPolicy) -> (f64, f64) {
    match policy {
        QualityPolicy::Strict => (7.0, 1.8),
        QualityPolicy::Balanced => (EM_SWEEP_COUNT_MIN_BALANCED, EM_RESONANCE_Q_MIN_BALANCED),
        QualityPolicy::Exploratory => (3.0, 0.75),
    }
}

pub(crate) fn breach_rate_greater_than(values: &[f64], threshold: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().filter(|value| **value > threshold).count() as f64 / values.len() as f64)
}

pub(crate) fn breach_rate_less_than(values: &[f64], threshold: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().filter(|value| **value < threshold).count() as f64 / values.len() as f64)
}
