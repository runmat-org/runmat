use serde::{Deserialize, Serialize};

use crate::refinement::{MeshRefinementOptions, RefinementStrategy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveConvergenceStatus {
    NotStarted,
    Disabled,
    Pending,
    Converged,
    MaxIterationsReached,
    ElementBudgetReached,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AdaptiveConvergenceMetrics {
    pub completed_iterations: usize,
    pub element_budget_reached: bool,
    pub previous_node_count: Option<usize>,
    pub current_node_count: Option<usize>,
    pub previous_element_count: Option<usize>,
    pub current_element_count: Option<usize>,
    pub field_change: Option<f64>,
    pub energy_change: Option<f64>,
    pub residual: Option<f64>,
}

pub fn evaluate_adaptive_convergence(
    options: &MeshRefinementOptions,
    metrics: AdaptiveConvergenceMetrics,
) -> AdaptiveConvergenceStatus {
    if matches!(
        options.strategy,
        RefinementStrategy::None | RefinementStrategy::Uniform
    ) {
        return AdaptiveConvergenceStatus::Disabled;
    }
    if metrics.element_budget_reached {
        return AdaptiveConvergenceStatus::ElementBudgetReached;
    }
    if metrics.completed_iterations >= options.max_iterations {
        return AdaptiveConvergenceStatus::MaxIterationsReached;
    }
    if metrics.completed_iterations > 0
        && matches!(
            (
                metrics.previous_node_count,
                metrics.current_node_count,
                metrics.previous_element_count,
                metrics.current_element_count,
            ),
            (Some(previous_nodes), Some(current_nodes), Some(previous_elements), Some(current_elements))
                if current_nodes <= previous_nodes && current_elements <= previous_elements
        )
    {
        return AdaptiveConvergenceStatus::Converged;
    }

    let mut considered_metric = false;
    let mut converged = true;

    if let Some(field_change) = metrics.field_change {
        considered_metric = true;
        converged &=
            field_change.is_finite() && field_change <= options.convergence.field_change_tolerance;
    }
    if let Some(energy_change) = metrics.energy_change {
        considered_metric = true;
        converged &= energy_change.is_finite()
            && energy_change <= options.convergence.energy_change_tolerance;
    }
    if let (Some(residual), Some(tolerance)) =
        (metrics.residual, options.convergence.residual_tolerance)
    {
        considered_metric = true;
        converged &= residual.is_finite() && residual <= tolerance;
    }

    if considered_metric && converged {
        AdaptiveConvergenceStatus::Converged
    } else {
        AdaptiveConvergenceStatus::Pending
    }
}
