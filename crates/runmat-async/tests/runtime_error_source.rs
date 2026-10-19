use runmat_async::{
    runtime_error, RuntimeError, RuntimeErrorOwner, RuntimeErrorSourceBound,
    SharedRuntimeErrorSource,
};
use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
type Owner = std::sync::Arc<()>;
#[cfg(target_arch = "wasm32")]
type Owner = std::rc::Rc<()>;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[derive(Debug)]
struct RetainedCause(Owner);

impl std::fmt::Display for RetainedCause {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("retained original cause")
    }
}
impl Error for RetainedCause {}

fn domain_wrapper(source: impl RuntimeErrorSourceBound) -> RuntimeError {
    runtime_error("domain wrapper")
        .with_identifier("RunMat:Test:Domain")
        .with_source(source)
        .build()
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn forwarding_bound_retains_nested_exact_source_and_local_owner_until_final_drop() {
    let owner = Owner::new(());
    let weak = Owner::downgrade(&owner);
    let inner = domain_wrapper(RetainedCause(owner));
    let outer = domain_wrapper(inner);
    assert_eq!(outer.identifier(), Some("RunMat:Test:Domain"));
    let inner = outer
        .source()
        .unwrap()
        .downcast_ref::<RuntimeError>()
        .unwrap();
    let leaf = inner
        .source()
        .unwrap()
        .downcast_ref::<RetainedCause>()
        .unwrap();
    assert_eq!(Owner::strong_count(&leaf.0), 1);
    assert_eq!(leaf.to_string(), "retained original cause");
    assert!(weak.upgrade().is_some());
    drop(outer);
    assert!(weak.upgrade().is_none());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn shared_rethrow_retains_the_original_and_its_downcastable_cause() {
    let owner = Owner::new(());
    let weak = Owner::downgrade(&owner);
    let original = RuntimeErrorOwner::new(
        runtime_error("source failure")
            .with_identifier("RunMat:Test:Original")
            .with_call_stack(vec!["origin".into()])
            .with_phase("execution")
            .with_gpu_gather_retry(runmat_async::GpuGatherRetry::Never)
            .with_failure_origin(runmat_async::RuntimeFailureOrigin::LanguageEvaluation)
            .with_source(RetainedCause(owner))
            .build(),
    );
    let caught = RuntimeErrorOwner::clone(&original);
    let reraised = RuntimeError::from_shared_original(caught);
    assert_eq!(reraised.identifier(), Some("RunMat:Test:Original"));
    assert_eq!(reraised.context.call_stack, ["origin"]);
    assert_eq!(reraised.context.phase.as_deref(), Some("execution"));
    assert_eq!(
        reraised.gpu_gather_retry,
        runmat_async::GpuGatherRetry::Never
    );
    assert_eq!(
        reraised.failure_origin,
        runmat_async::RuntimeFailureOrigin::LanguageEvaluation
    );
    let shared = reraised
        .source()
        .unwrap()
        .downcast_ref::<SharedRuntimeErrorSource>()
        .unwrap();
    assert!(std::ptr::eq(shared.original(), original.as_ref()));
    assert!(shared.original().source().unwrap().is::<RetainedCause>());
    drop(reraised);
    drop(original);
    assert!(weak.upgrade().is_none());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_source_bound_and_runtime_error_still_require_thread_safe_transport() {
    fn send_sync<T: Send + Sync + ?Sized>() {}
    fn source_bound_implies_transport<T: RuntimeErrorSourceBound>() {
        send_sync::<T>();
    }
    source_bound_implies_transport::<std::io::Error>();
    source_bound_implies_transport::<RuntimeError>();
    send_sync::<RuntimeError>();
    let error = domain_wrapper(std::io::Error::other("native cause"));
    assert!(error.source().unwrap().is::<std::io::Error>());
}
