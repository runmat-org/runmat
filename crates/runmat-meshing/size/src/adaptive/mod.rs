mod convergence;
mod default_indicators;
mod indicator_plan;
mod iteration;
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
pub use iteration::{AdaptiveIterationSummary, SizingFieldUpdate};
pub use markers::{
    build_refinement_markers_from_samples, RefinementIndicatorSample, RefinementMarker,
    RefinementMarkerError, RefinementMarkerOptions,
};

#[cfg(test)]
mod tests;
