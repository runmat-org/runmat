use crate::{
    adaptive::{
        evaluate_adaptive_convergence, AdaptiveConvergenceMetrics, AdaptiveConvergenceStatus,
    },
    refinement::{MeshRefinementOptions, RefinementStrategy},
};

#[test]
fn adaptive_convergence_is_disabled_for_nonadaptive_strategies() {
    let mut options = MeshRefinementOptions {
        strategy: RefinementStrategy::None,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Disabled
    );

    options.strategy = RefinementStrategy::Uniform;
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Disabled
    );
}

#[test]
fn adaptive_convergence_reports_hard_stop_statuses_first() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        max_iterations: 2,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                element_budget_reached: true,
                completed_iterations: 2,
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::ElementBudgetReached
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                completed_iterations: 2,
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::MaxIterationsReached
    );
}

#[test]
fn adaptive_convergence_requires_finite_metric_within_tolerance() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.05),
                energy_change: Some(0.02),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.051),
                energy_change: Some(0.02),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(f64::NAN),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
    assert_eq!(
        evaluate_adaptive_convergence(&options, AdaptiveConvergenceMetrics::default()),
        AdaptiveConvergenceStatus::Pending
    );
}

#[test]
fn adaptive_convergence_detects_no_topology_growth() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                completed_iterations: 1,
                previous_node_count: Some(24),
                current_node_count: Some(24),
                previous_element_count: Some(48),
                current_element_count: Some(48),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                completed_iterations: 1,
                previous_node_count: Some(24),
                current_node_count: Some(25),
                previous_element_count: Some(48),
                current_element_count: Some(49),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
}

#[test]
fn adaptive_convergence_residual_metric_is_opt_in() {
    let mut options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                residual: Some(1.0e6),
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );

    options.convergence.residual_tolerance = Some(1.0e-6);
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                residual: Some(1.0e-7),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                residual: Some(1.0e-5),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
}
