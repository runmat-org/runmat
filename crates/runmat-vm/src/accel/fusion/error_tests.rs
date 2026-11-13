use super::{classify_provider_error, fusion_execution_error, FusionAttemptError};
use runmat_accelerate_api::ProviderExecutionError;
use runmat_runtime::runtime_error::{build_runtime_error, GpuGatherRetry};
use std::error::Error;
use std::fmt;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(target_arch = "wasm32")]
use std::rc::Rc as Owner;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc as Owner;

#[derive(Debug)]
struct RetainedFailure(Owner<()>);

impl fmt::Display for RetainedFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("retained fusion failure")
    }
}
impl Error for RetainedFailure {}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fusion_publication_preserves_language_error_and_local_owner() {
    let owner = Owner::new(());
    let weak = Owner::downgrade(&owner);
    let language = build_runtime_error("placement refused execution")
        .with_identifier("RunMat:PlacementRefused")
        .with_source(RetainedFailure(owner))
        .build();
    let result = fusion_execution_error(ProviderExecutionError::from_source(language));
    assert_eq!(result.message(), "placement refused execution");
    assert_eq!(result.identifier(), Some("RunMat:PlacementRefused"));
    assert_eq!(result.gpu_gather_retry, GpuGatherRetry::Never);
    let mut source: &(dyn Error + 'static) = &result;
    loop {
        if let Some(retained) = source.downcast_ref::<RetainedFailure>() {
            assert!(Owner::ptr_eq(&retained.0, &weak.upgrade().unwrap()));
            break;
        }
        source = source
            .source()
            .expect("original typed failure must survive");
    }
    assert!(weak.upgrade().is_some());
    drop(result);
    assert!(weak.upgrade().is_none());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fusion_replays_only_typed_unsupported_before_effect() {
    let unsupported = || {
        ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
            "no fused kernel",
        ))
    };
    assert!(matches!(
        classify_provider_error(unsupported()),
        FusionAttemptError::Deopt(_)
    ));
    assert!(matches!(
        classify_provider_error(unsupported().after_effect()),
        FusionAttemptError::Terminal(_)
    ));
    let hard = build_runtime_error("provider cancelled")
        .with_identifier("RunMat:ExecutionCancelled")
        .build();
    match classify_provider_error(ProviderExecutionError::from_source(hard)) {
        FusionAttemptError::Terminal(error) => {
            assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
            assert_eq!(error.message(), "provider cancelled");
        }
        FusionAttemptError::Deopt(_) => panic!("hard provider errors must not replay"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_fusion_error_bounds_are_unchanged() {
    fn assert_thread_safe<T: Send + Sync>() {}
    assert_thread_safe::<ProviderExecutionError>();
    assert_thread_safe::<runmat_runtime::RuntimeError>();
}
