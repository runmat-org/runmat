//! Original invocation ownership must not be replaced by ambient session state.
use super::*;
use futures::executor::block_on;
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::rc::Rc;

fn runtime(source: &CancellationSource) -> crate::context::RuntimeContext {
    crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    )
}
#[test]
fn explicit_numeric_gather_uses_original_context_in_both_cancellation_directions() {
    let source = CancellationSource::new();
    let foreign_source = CancellationSource::new();
    let original = runtime(&source);
    original.set_error_namespace("OriginalGather");
    let foreign = runtime(&foreign_source);
    foreign.set_error_namespace("ForeignGather");
    foreign_source.cancel(CancellationReason::User);
    {
        let _foreign = foreign.enter();
        let mut cooperation = ScalarCooperation::for_builtin(
            runmat_builtins::CONEPROG_CATALOG_ENTRY.identity,
            Some(&original),
        );
        assert_eq!(
            block_on(numeric::vector(
                "f",
                Value::Num(3.0),
                FiniteMode::Finite,
                &mut cooperation
            ))
            .unwrap(),
            vec![3.0]
        );
    }
    source.cancel(CancellationReason::User);
    let unrelated = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _unrelated = unrelated.enter();
    let cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::CONEPROG_CATALOG_ENTRY.identity,
        Some(&original),
    );
    let error = block_on(numeric::gather(Value::Num(3.0), &cooperation)).unwrap_err();
    assert_eq!(error.identifier(), Some("OriginalGather:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}
