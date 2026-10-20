use super::*;
use crate::context::{RuntimeContext, RuntimeServicePorts};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::rc::Rc;
use std::sync::Arc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_finite_calls_precede_missing_resource_evidence_without_file_effects() {
    let _lock = runmat_filesystem::provider_override_lock();
    let fs = Arc::new(CountedFs::new("/audio"));
    let _guard = runmat_filesystem::replace_provider(fs.clone());
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(cancellation.token())
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(Budget::new(Some(1024))))
                .with_placement(Rc::new(Placement::default())),
        );
    cancellation.cancel(CancellationReason::User);
    for error in [
        block_on(audioread_builtin(
            Some(runtime.clone()),
            Value::from("/audio/tone.wav"),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("cancelled decode before resource lookup"),
        block_on(audioinfo_builtin(
            Some(runtime.clone()),
            Value::from("/audio/tone.wav"),
        ))
        .expect_err("cancelled metadata before resource lookup"),
    ] {
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    }
    let invalid = block_on(audioread_builtin(
        Some(runtime),
        Value::Num(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("pure argument admission retains precedence");
    assert_eq!(
        invalid.identifier(),
        Some("RunMat:audioread:InvalidArgument")
    );
    assert_eq!(fs.effects(), (0, 0, 0));
}
