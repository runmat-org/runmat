use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use runmat_meshing_core::contracts::AnalysisMeshArtifact;
use runmat_meshing_size::adaptive::{AdaptiveConvergenceStatus, RefinementIndicatorStatus};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshAdaptiveEvidence {
    pub iteration_count: usize,
    #[serde(default)]
    pub latest_iteration_index: Option<usize>,
    #[serde(default)]
    pub latest_convergence_status: Option<String>,
    #[serde(default)]
    pub latest_indicator_count: usize,
    #[serde(default)]
    pub latest_used_indicator_count: usize,
    #[serde(default)]
    pub latest_marker_count: usize,
    #[serde(default)]
    pub latest_sizing_update_sample_count: usize,
    #[serde(default)]
    pub marker_count: usize,
    #[serde(default)]
    pub sizing_update_sample_count: usize,
    #[serde(default)]
    pub latest_indicator_status_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub latest_marker_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub latest_sizing_update_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub marker_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub sizing_update_by_reason: BTreeMap<String, usize>,
}

pub(crate) fn adaptive_evidence(mesh: &AnalysisMeshArtifact) -> MeshAdaptiveEvidence {
    let mut marker_count = 0_usize;
    let mut sizing_update_sample_count = 0_usize;
    let mut marker_by_reason = BTreeMap::<String, usize>::new();
    let mut sizing_update_by_reason = BTreeMap::<String, usize>::new();
    for iteration in &mesh.adaptive_iterations {
        marker_count += iteration.markers.len();
        sizing_update_sample_count += iteration.sizing_update.samples.len();
        for marker in &iteration.markers {
            *marker_by_reason.entry(marker.reason.clone()).or_default() += 1;
        }
        for sample in &iteration.sizing_update.samples {
            let reason = sample
                .reason
                .clone()
                .unwrap_or_else(|| "unspecified".to_string());
            *sizing_update_by_reason.entry(reason).or_default() += 1;
        }
    }

    let Some(latest) = mesh.adaptive_iterations.last() else {
        return MeshAdaptiveEvidence {
            iteration_count: 0,
            ..MeshAdaptiveEvidence::default()
        };
    };

    let mut latest_indicator_status_counts = BTreeMap::<String, usize>::new();
    for indicator in &latest.indicators {
        *latest_indicator_status_counts
            .entry(indicator_status_label(indicator.status))
            .or_default() += 1;
    }
    let mut latest_marker_by_reason = BTreeMap::<String, usize>::new();
    for marker in &latest.markers {
        *latest_marker_by_reason
            .entry(marker.reason.clone())
            .or_default() += 1;
    }
    let mut latest_sizing_update_by_reason = BTreeMap::<String, usize>::new();
    for sample in &latest.sizing_update.samples {
        let reason = sample
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *latest_sizing_update_by_reason.entry(reason).or_default() += 1;
    }

    MeshAdaptiveEvidence {
        iteration_count: mesh.adaptive_iterations.len(),
        latest_iteration_index: Some(latest.iteration_index),
        latest_convergence_status: Some(convergence_status_label(latest.convergence_status)),
        latest_indicator_count: latest.indicators.len(),
        latest_used_indicator_count: latest
            .indicators
            .iter()
            .filter(|indicator| indicator.status == RefinementIndicatorStatus::Used)
            .count(),
        latest_marker_count: latest.markers.len(),
        latest_sizing_update_sample_count: latest.sizing_update.samples.len(),
        marker_count,
        sizing_update_sample_count,
        latest_indicator_status_counts,
        latest_marker_by_reason,
        latest_sizing_update_by_reason,
        marker_by_reason,
        sizing_update_by_reason,
    }
}

fn convergence_status_label(status: AdaptiveConvergenceStatus) -> String {
    match status {
        AdaptiveConvergenceStatus::NotStarted => "not_started",
        AdaptiveConvergenceStatus::Disabled => "disabled",
        AdaptiveConvergenceStatus::Pending => "pending",
        AdaptiveConvergenceStatus::Converged => "converged",
        AdaptiveConvergenceStatus::MaxIterationsReached => "max_iterations_reached",
        AdaptiveConvergenceStatus::ElementBudgetReached => "element_budget_reached",
    }
    .to_string()
}

fn indicator_status_label(status: RefinementIndicatorStatus) -> String {
    match status {
        RefinementIndicatorStatus::Used => "used",
        RefinementIndicatorStatus::SkippedMissingField => "skipped_missing_field",
        RefinementIndicatorStatus::SkippedNotApplicable => "skipped_not_applicable",
        RefinementIndicatorStatus::SkippedBudget => "skipped_budget",
        RefinementIndicatorStatus::SkippedQuality => "skipped_quality",
    }
    .to_string()
}
