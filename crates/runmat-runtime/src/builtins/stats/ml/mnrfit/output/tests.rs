use super::*;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::rc::Rc;

fn cancellation_after(probes: &Cell<usize>, source: &CancellationSource, limit: usize) {
    let current = probes.get() + 1;
    probes.set(current);
    if current == limit {
        source.cancel(CancellationReason::User);
    }
}

#[test]
fn cancellation_during_large_coefficient_copy_prevents_publication() {
    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let probes = Cell::new(0);
    let coefficients = vec![1.0; 40_000];
    let error = block_on(coefficients_value(
        &coefficients,
        40_000,
        1,
        &|| {
            cancellation_after(&probes, &source, 3);
            check_cancellation(Some(&runtime))
        },
        &mut FitCooperation::new(None),
    ))
    .expect_err("cancelled output copy cannot return coefficients");
    assert!(probes.get() >= 3);
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}

#[test]
fn cancellation_during_correlation_output_prevents_stats_publication() {
    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let probes = Cell::new(0);
    let dim = 64;
    let covariance = vec![1.0; dim * dim];
    let errors = vec![1.0; dim];
    let error = block_on(coeffcorr_value(
        &covariance,
        &errors,
        &|| {
            cancellation_after(&probes, &source, 4);
            check_cancellation(Some(&runtime))
        },
        &mut FitCooperation::new(None),
    ))
    .expect_err("cancelled correlation matrix cannot be published");
    assert!(probes.get() >= 4);
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}
