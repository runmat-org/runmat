use super::error::resource_category;
use super::*;

#[test]
fn resource_failures_keep_their_specific_budget_category() {
    assert_eq!(
        resource_category("geometry evaluation IterationBudgetExceeded"),
        MeshingFailureCategory::IterationBudgetExceeded
    );
    assert_eq!(
        resource_category("geometry evaluation AllocationBudgetExceeded"),
        MeshingFailureCategory::MemoryBudgetExceeded
    );
    assert_eq!(
        resource_category("boundary face inventory exceeds its hard limit"),
        MeshingFailureCategory::ElementBudgetExceeded
    );
    assert_eq!(
        resource_category("curved-node candidate work exceeds its hard limit"),
        MeshingFailureCategory::SearchWorkBudgetExceeded
    );
}

#[test]
fn solver_checkpoint_cadence_respects_the_kernel_latency_bound() {
    let mut request = crate::serial_tests::request();
    request.cancellation.maximum_work_units_between_checks = u64::MAX;
    let options = solver_options(&request);
    assert_eq!(
        options.cancellation_check_interval,
        DelaunaySolverTopologyOptions::default().cancellation_check_interval
    );
    assert_eq!(
        options.maximum_curved_optimization_sweeps,
        DelaunaySolverTopologyOptions::default().maximum_curved_optimization_sweeps
    );
}
