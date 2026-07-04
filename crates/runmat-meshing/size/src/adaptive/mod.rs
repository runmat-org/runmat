use serde::{Deserialize, Serialize};

use crate::field::{MeshSizingField, SizingSample};

mod convergence;
mod default_indicators;
mod indicator_plan;
mod markers;

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
pub use markers::{
    build_refinement_markers_from_samples, RefinementIndicatorSample, RefinementMarker,
    RefinementMarkerError, RefinementMarkerOptions,
};

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
