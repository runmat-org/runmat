use serde::{Deserialize, Serialize};

use crate::field::{MeshSizingField, SizingSample};

mod convergence;
mod default_indicators;
mod indicator_plan;

pub use convergence::{
    evaluate_adaptive_convergence, AdaptiveConvergenceMetrics, AdaptiveConvergenceStatus,
};
pub use default_indicators::{
    default_refinement_indicators_for_analysis, structural_static_default_refinement_indicators,
};
pub use indicator_plan::{
    plan_refinement_indicators, RefinementIndicatorAvailability, RefinementIndicatorKey,
    RefinementIndicatorStatus, RefinementIndicatorSummary,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementMarker {
    pub entity_id: String,
    pub weight: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementIndicatorSample {
    pub entity_id: String,
    pub position_m: [f64; 3],
    pub indicator_value: f64,
    pub current_size_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RefinementMarkerOptions {
    pub max_markers: usize,
    pub min_relative_value: f64,
    pub target_size_scale: f64,
}

impl Default for RefinementMarkerOptions {
    fn default() -> Self {
        Self {
            max_markers: 64,
            min_relative_value: 0.25,
            target_size_scale: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementMarkerError {
    InvalidMaxMarkers,
    InvalidMinRelativeValue,
    InvalidTargetSizeScale,
}

pub fn build_refinement_markers_from_samples(
    samples: &[RefinementIndicatorSample],
    reason: &str,
    options: RefinementMarkerOptions,
) -> Result<(Vec<RefinementMarker>, SizingFieldUpdate), RefinementMarkerError> {
    if options.max_markers == 0 {
        return Err(RefinementMarkerError::InvalidMaxMarkers);
    }
    if !options.min_relative_value.is_finite() || !(0.0..=1.0).contains(&options.min_relative_value)
    {
        return Err(RefinementMarkerError::InvalidMinRelativeValue);
    }
    if !options.target_size_scale.is_finite()
        || options.target_size_scale <= 0.0
        || options.target_size_scale >= 1.0
    {
        return Err(RefinementMarkerError::InvalidTargetSizeScale);
    }

    let mut finite_samples = samples
        .iter()
        .filter(|sample| {
            sample.indicator_value.is_finite()
                && sample.indicator_value > 0.0
                && sample.current_size_m.is_finite()
                && sample.current_size_m > 0.0
                && sample.position_m.iter().all(|value| value.is_finite())
        })
        .collect::<Vec<_>>();
    let Some(max_value) = finite_samples
        .iter()
        .map(|sample| sample.indicator_value)
        .reduce(f64::max)
    else {
        return Ok((Vec::new(), SizingFieldUpdate::default()));
    };

    finite_samples
        .retain(|sample| sample.indicator_value / max_value >= options.min_relative_value);
    finite_samples.sort_by(|left, right| {
        right
            .indicator_value
            .total_cmp(&left.indicator_value)
            .then_with(|| left.entity_id.cmp(&right.entity_id))
    });
    finite_samples.truncate(options.max_markers);

    let markers = finite_samples
        .iter()
        .map(|sample| RefinementMarker {
            entity_id: sample.entity_id.clone(),
            weight: sample.indicator_value / max_value,
            reason: reason.to_string(),
        })
        .collect::<Vec<_>>();
    let sizing_update = SizingFieldUpdate {
        samples: finite_samples
            .into_iter()
            .map(|sample| SizingSample {
                position_m: sample.position_m,
                target_size_m: sample.current_size_m * options.target_size_scale,
                reason: Some(reason.to_string()),
            })
            .collect(),
        min_size_m: None,
        max_size_m: None,
    };

    Ok((markers, sizing_update))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SizingFieldUpdate {
    #[serde(default)]
    pub samples: Vec<SizingSample>,
    #[serde(default)]
    pub min_size_m: Option<f64>,
    #[serde(default)]
    pub max_size_m: Option<f64>,
}

impl SizingFieldUpdate {
    pub fn apply_to(self, sizing: &mut MeshSizingField) {
        if let Some(min_size_m) = self.min_size_m {
            sizing.min_size_m = Some(match sizing.min_size_m {
                Some(existing) => existing.min(min_size_m),
                None => min_size_m,
            });
        }
        if let Some(max_size_m) = self.max_size_m {
            sizing.max_size_m = Some(match sizing.max_size_m {
                Some(existing) => existing.max(max_size_m),
                None => max_size_m,
            });
        }
        sizing.samples.extend(self.samples);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveIterationSummary {
    pub iteration_index: usize,
    pub node_count: usize,
    pub element_count: usize,
    pub convergence_status: AdaptiveConvergenceStatus,
    #[serde(default)]
    pub indicators: Vec<RefinementIndicatorSummary>,
    #[serde(default)]
    pub markers: Vec<RefinementMarker>,
    #[serde(default)]
    pub sizing_update: SizingFieldUpdate,
}

#[cfg(test)]
mod tests;
