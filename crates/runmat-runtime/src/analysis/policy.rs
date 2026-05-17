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

pub(crate) fn electromagnetic_thresholds_for_policy(
    policy: QualityPolicy,
) -> (
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
) {
    match policy {
        QualityPolicy::Strict => (
            1.5, 0.12, 0.95, 0.05, 0.45, 8.0e3, 0.85, 0.95, 0.9, 0.55, 0.35, 0.7, 0.9, 0.85, 0.4,
            0.18, 0.25, 0.25, 0.35, 0.80, 0.18, 0.50,
        ),
        QualityPolicy::Balanced => (
            EM_CONDUCTIVITY_SPREAD_THRESHOLD_BALANCED,
            EM_HETEROGENEITY_THRESHOLD_BALANCED,
            EM_ASSIGNMENT_COVERAGE_MIN_BALANCED,
            EM_FALLBACK_COEFFICIENT_MAX_BALANCED,
            EM_REGION_CONTRAST_MAX_BALANCED,
            EM_CONDITIONING_MAX_BALANCED,
            EM_SOURCE_REALIZATION_MIN_BALANCED,
            EM_SOURCE_REGION_COVERAGE_MIN_BALANCED,
            EM_SOURCE_MATERIAL_ALIGNMENT_MIN_BALANCED,
            EM_SOURCE_OVERLAP_MAX_BALANCED,
            EM_SOURCE_INTERFERENCE_MAX_BALANCED,
            EM_BOUNDARY_ANCHOR_MIN_BALANCED,
            EM_BOUNDARY_LOCALIZATION_MIN_BALANCED,
            EM_GROUND_EFFECTIVENESS_MIN_BALANCED,
            EM_INSULATION_LEAKAGE_MAX_BALANCED,
            EM_FLUX_DIVERGENCE_MAX_BALANCED,
            EM_ENERGY_IMBALANCE_MAX_BALANCED,
            EM_BOUNDARY_ENERGY_MIN_BALANCED,
            EM_BOUNDARY_PENALTY_CONTRIBUTION_MAX_BALANCED,
            EM_SOURCE_REGION_ENERGY_CONSISTENCY_MIN_BALANCED,
            EM_REAL_RESIDUAL_MAX_BALANCED,
            EM_IMAG_RESIDUAL_MAX_BALANCED,
        ),
        QualityPolicy::Exploratory => (
            3.0, 0.35, 0.5, 0.65, 1.8, 1.5e5, 0.2, 0.35, 0.25, 0.95, 0.85, 0.15, 0.4, 0.35, 0.9,
            0.8, 1.2, 0.03, 0.9, 0.25, 0.85, 1.2,
        ),
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
