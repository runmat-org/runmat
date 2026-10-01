use super::*;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn cancellation_after_solver_has_started_prevents_fit_publication() {
    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let data = PreparedData {
        design: vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0],
        rows: 4,
        cols: 2,
        outcomes: vec![vec![0.0, 0.0, 1.0, 1.0]],
        weights: vec![1.0; 4],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        observations: 4.0,
    };
    let probes = Cell::new(0);
    let error = block_on(fit_nominal_logit(
        &data,
        100,
        1.0e-8,
        &|| {
            let current = probes.get() + 1;
            probes.set(current);
            if current == 6 {
                source.cancel(CancellationReason::User);
            }
            check_cancellation(Some(&runtime))
        },
        &mut FitCooperation::new(None),
    ))
    .expect_err("mid-fit cancellation must not publish a fit");
    assert!(probes.get() >= 6);
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}
