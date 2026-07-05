use serde::{Deserialize, Serialize};

pub const MODULE_PURPOSE: &str = "interior and CAD-constrained smoothing after constraint recovery";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronSmoothingOptions {
    pub min_volume_m3: f64,
    pub min_scaled_jacobian: f64,
    pub min_scaled_jacobian_improvement: f64,
    pub max_aspect_ratio_growth: f64,
}

impl Default for TetrahedronSmoothingOptions {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            min_scaled_jacobian: 0.15,
            min_scaled_jacobian_improvement: 1.0e-12,
            max_aspect_ratio_growth: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronSmoothingQuality {
    pub tetrahedron_id: u32,
    pub volume_m3: f64,
    pub scaled_jacobian: f64,
    pub aspect_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronSmoothingRejectionReason {
    EmptyPatch,
    NonFiniteQuality,
    VolumeBelowThreshold,
    ScaledJacobianBelowThreshold,
    QualityDoesNotImprove,
    AspectRatioRegressed,
}

impl TetrahedronSmoothingRejectionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmptyPatch => "empty_patch",
            Self::NonFiniteQuality => "non_finite_quality",
            Self::VolumeBelowThreshold => "volume_below_threshold",
            Self::ScaledJacobianBelowThreshold => "scaled_jacobian_below_threshold",
            Self::QualityDoesNotImprove => "quality_does_not_improve",
            Self::AspectRatioRegressed => "aspect_ratio_regressed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronSmoothingEvaluation {
    pub accepted: bool,
    pub initial_min_scaled_jacobian: f64,
    pub candidate_min_scaled_jacobian: f64,
    pub initial_max_aspect_ratio: f64,
    pub candidate_max_aspect_ratio: f64,
    #[serde(default)]
    pub rejection_reason: Option<TetrahedronSmoothingRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronSmoothingError {
    InvalidOptions,
}

pub fn evaluate_tetrahedron_smoothing_candidate(
    current: &[TetrahedronSmoothingQuality],
    candidate: &[TetrahedronSmoothingQuality],
    options: TetrahedronSmoothingOptions,
) -> Result<TetrahedronSmoothingEvaluation, TetrahedronSmoothingError> {
    validate_options(options)?;
    let current_summary = summarize_quality(current);
    let candidate_summary = summarize_quality(candidate);
    let rejection_reason = smoothing_rejection_reason(current_summary, candidate_summary, options);

    Ok(TetrahedronSmoothingEvaluation {
        accepted: rejection_reason.is_none(),
        initial_min_scaled_jacobian: current_summary.min_scaled_jacobian,
        candidate_min_scaled_jacobian: candidate_summary.min_scaled_jacobian,
        initial_max_aspect_ratio: current_summary.max_aspect_ratio,
        candidate_max_aspect_ratio: candidate_summary.max_aspect_ratio,
        rejection_reason,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TetrahedronSmoothingQualitySummary {
    count: usize,
    min_volume_m3: f64,
    min_scaled_jacobian: f64,
    max_aspect_ratio: f64,
    finite: bool,
}

fn summarize_quality(
    tetrahedra: &[TetrahedronSmoothingQuality],
) -> TetrahedronSmoothingQualitySummary {
    let mut summary = TetrahedronSmoothingQualitySummary {
        count: tetrahedra.len(),
        min_volume_m3: f64::INFINITY,
        min_scaled_jacobian: f64::INFINITY,
        max_aspect_ratio: 0.0,
        finite: true,
    };
    for tetrahedron in tetrahedra {
        summary.finite &= tetrahedron.volume_m3.is_finite()
            && tetrahedron.scaled_jacobian.is_finite()
            && tetrahedron.aspect_ratio.is_finite();
        summary.min_volume_m3 = summary.min_volume_m3.min(tetrahedron.volume_m3);
        summary.min_scaled_jacobian = summary.min_scaled_jacobian.min(tetrahedron.scaled_jacobian);
        summary.max_aspect_ratio = summary.max_aspect_ratio.max(tetrahedron.aspect_ratio);
    }
    if tetrahedra.is_empty() {
        summary.min_volume_m3 = 0.0;
        summary.min_scaled_jacobian = 0.0;
    }
    summary
}

fn smoothing_rejection_reason(
    current: TetrahedronSmoothingQualitySummary,
    candidate: TetrahedronSmoothingQualitySummary,
    options: TetrahedronSmoothingOptions,
) -> Option<TetrahedronSmoothingRejectionReason> {
    if current.count == 0 || candidate.count == 0 {
        return Some(TetrahedronSmoothingRejectionReason::EmptyPatch);
    }
    if !current.finite || !candidate.finite {
        return Some(TetrahedronSmoothingRejectionReason::NonFiniteQuality);
    }
    if candidate.min_volume_m3 < options.min_volume_m3 {
        return Some(TetrahedronSmoothingRejectionReason::VolumeBelowThreshold);
    }
    if candidate.min_scaled_jacobian < options.min_scaled_jacobian {
        return Some(TetrahedronSmoothingRejectionReason::ScaledJacobianBelowThreshold);
    }
    if candidate.min_scaled_jacobian
        <= current.min_scaled_jacobian + options.min_scaled_jacobian_improvement
    {
        return Some(TetrahedronSmoothingRejectionReason::QualityDoesNotImprove);
    }
    if candidate.max_aspect_ratio > current.max_aspect_ratio + options.max_aspect_ratio_growth {
        return Some(TetrahedronSmoothingRejectionReason::AspectRatioRegressed);
    }
    None
}

fn validate_options(options: TetrahedronSmoothingOptions) -> Result<(), TetrahedronSmoothingError> {
    if !options.min_volume_m3.is_finite()
        || options.min_volume_m3 < 0.0
        || !options.min_scaled_jacobian.is_finite()
        || options.min_scaled_jacobian < 0.0
        || !options.min_scaled_jacobian_improvement.is_finite()
        || options.min_scaled_jacobian_improvement < 0.0
        || !options.max_aspect_ratio_growth.is_finite()
        || options.max_aspect_ratio_growth < 0.0
    {
        return Err(TetrahedronSmoothingError::InvalidOptions);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_candidate_that_improves_minimum_scaled_jacobian() {
        let current = [quality(1, 1.0, 0.2, 3.0), quality(2, 1.0, 0.4, 2.0)];
        let candidate = [quality(1, 1.0, 0.3, 2.5), quality(2, 1.0, 0.45, 2.0)];

        let evaluation = evaluate_tetrahedron_smoothing_candidate(
            &current,
            &candidate,
            TetrahedronSmoothingOptions {
                min_scaled_jacobian: 0.15,
                ..TetrahedronSmoothingOptions::default()
            },
        )
        .expect("valid smoothing options should evaluate");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.rejection_reason, None);
        assert!(evaluation.candidate_min_scaled_jacobian > evaluation.initial_min_scaled_jacobian);
    }

    #[test]
    fn rejects_candidate_that_does_not_improve_quality() {
        let current = [quality(1, 1.0, 0.3, 2.0)];
        let candidate = [quality(1, 1.0, 0.3, 2.0)];

        let evaluation = evaluate_tetrahedron_smoothing_candidate(
            &current,
            &candidate,
            TetrahedronSmoothingOptions::default(),
        )
        .expect("valid smoothing options should evaluate");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(TetrahedronSmoothingRejectionReason::QualityDoesNotImprove)
        );
    }

    fn quality(
        tetrahedron_id: u32,
        volume_m3: f64,
        scaled_jacobian: f64,
        aspect_ratio: f64,
    ) -> TetrahedronSmoothingQuality {
        TetrahedronSmoothingQuality {
            tetrahedron_id,
            volume_m3,
            scaled_jacobian,
            aspect_ratio,
        }
    }
}
