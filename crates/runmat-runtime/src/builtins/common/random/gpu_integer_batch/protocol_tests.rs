mod support;

use super::execute_selected;
use futures::executor::block_on;
use runmat_accelerate_api::random::GpuBoundedIntegerDomain;
use runmat_accelerate_api::NumericElementType;
use runmat_execution::cancellation::CancellationSource;
use std::future::Future;
use std::rc::Rc;
use std::task::{Context, Poll};
use support::{runtime, selected_first, Placement, ScriptedProvider};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn coordinator_acknowledges_pending_batches_before_publishing_completion() {
    let provider = ScriptedProvider::new(None);
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let placement = Rc::new(Placement::default());
    let cancellation = CancellationSource::new();
    let _runtime =
        crate::context::RuntimeContextGuard::enter(runtime(placement.clone(), &cancellation));
    let source = provider.source();
    let shape = [1, 4];
    let domain = GpuBoundedIntegerDomain::new(-2, 7).expect("non-power-of-two domain");
    let first = selected_first(&source.handle, domain, &shape);

    let output = block_on(execute_selected(
        &source.handle,
        domain,
        &shape,
        NumericElementType::I32,
        first,
    ))
    .expect("three acknowledged batches complete");

    assert_eq!(provider.attempts(), [0, 8, 16]);
    assert_eq!(provider.acknowledgements(), 3);
    assert_eq!(placement.observations.get(), 3);
    assert_eq!(output.shape(), shape);
    assert_eq!(output.physical().element_type(), NumericElementType::I32);
    assert_eq!(super::super::gpu::snapshot().unwrap().position, 4);
    assert_eq!(
        provider.provisional_retirements(),
        [0, 1],
        "only the unpublished completion mask retires"
    );
    drop(output);
    assert_eq!(provider.provisional_retirements(), [1, 1]);
    assert_eq!(source.probe.retirements(), 0, "prototype remains borrowed");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dropping_cancelled_coordinator_retires_unpublished_pending_state() {
    let provider = ScriptedProvider::new(Some(2));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let placement = Rc::new(Placement::default());
    let cancellation = CancellationSource::new();
    let _runtime =
        crate::context::RuntimeContextGuard::enter(runtime(placement.clone(), &cancellation));
    let source = provider.source();
    let shape = [1, 4];
    let domain = GpuBoundedIntegerDomain::new(-2, 7).expect("non-power-of-two domain");
    let first = selected_first(&source.handle, domain, &shape);
    let mut execution = Box::pin(execute_selected(
        &source.handle,
        domain,
        &shape,
        NumericElementType::I32,
        first,
    ));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        execution.as_mut().poll(&mut context),
        Poll::Pending
    ));
    assert_eq!(provider.attempts(), [0, 8]);
    assert_eq!(provider.acknowledgements(), 1);
    assert_eq!(placement.observations.get(), 1);
    assert_eq!(super::super::gpu::snapshot().unwrap().position, 4);

    assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
    drop(execution);

    assert_eq!(provider.provisional_retirements(), [1, 1]);
    assert_eq!(source.probe.retirements(), 0, "prototype remains borrowed");
}
