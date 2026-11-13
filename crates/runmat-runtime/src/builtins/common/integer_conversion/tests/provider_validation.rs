mod fixtures;
use super::super::cast_value;
use crate::builtins::common::test_support;
use fixtures::{input, provider, ResultMode};
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerTensorView, HostTensorView, IntegerElementType, ThreadProviderGuard,
};
use runmat_types::IntegerClass;
use runmat_value::Value;
use std::sync::atomic::Ordering;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn abandoning_pending_cast_restores_borrowed_metadata_and_allocation() {
    use futures::{task::noop_waker, FutureExt};
    use std::task::{Context, Poll};
    let _lock = test_support::accel_test_lock();
    let provider = provider(ResultMode::Pending);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = input(provider);
    let original_transpose = runmat_accelerate_api::handle_transpose_info(&input);
    let mut call = Box::pin(cast_value(
        Value::GpuTensor(input.clone()),
        IntegerClass::Int8,
    ));
    let waker = noop_waker();
    assert!(matches!(
        call.poll_unpin(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(!runmat_accelerate_api::handle_is_logical(&input));
    drop(call);
    assert!(!runmat_accelerate_api::handle_is_logical(&input));
    assert_eq!(
        runmat_accelerate_api::handle_transpose_info(&input),
        original_transpose
    );
    assert!(provider.created.lock().unwrap().is_empty());
    assert!(block_on(provider.inner.download(&input)).is_ok());
    let physical = input.allocation_lease().downgrade();
    drop(input);
    assert!(physical.acquire().is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn malformed_fresh_outputs_are_freed_without_touching_borrowed_input() {
    let _lock = test_support::accel_test_lock();
    for mode in [
        ResultMode::WrongClass,
        ResultMode::ForeignOwner,
        ResultMode::Alias,
    ] {
        let provider = provider(mode);
        let _current = ThreadProviderGuard::set(Some(provider));
        let input = input(provider);
        assert!(block_on(cast_value(
            Value::GpuTensor(input.clone()),
            IntegerClass::Int8
        ))
        .is_err());
        let created = provider.created.lock().unwrap();
        assert_eq!(
            created.len(),
            usize::from(!matches!(mode, ResultMode::Alias))
        );
        assert!(created.iter().all(|output| output.acquire().is_none()));
        assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
        assert!(block_on(provider.inner.download(&input)).is_ok());
        let physical = input.allocation_lease().downgrade();
        drop(input);
        assert!(physical.acquire().is_none());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn unavailable_input_owner_is_rejected_before_hook_or_download() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(ResultMode::Failure);
    let _current = ThreadProviderGuard::set(Some(provider));
    let foreign = InProcessProvider::new();
    let input = foreign
        .upload(&HostTensorView {
            data: &[2.0],
            shape: &[1, 1],
        })
        .unwrap();
    assert!(block_on(cast_value(
        Value::GpuTensor(input.clone()),
        IntegerClass::Int8
    ))
    .is_err());
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert!(block_on(foreign.download(&input)).is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn only_typed_unsupported_permits_owner_download_fallback() {
    let _lock = test_support::accel_test_lock();
    for mode in [ResultMode::Failure, ResultMode::Unsupported] {
        let provider = provider(mode);
        let _current = ThreadProviderGuard::set(Some(provider));
        let input = input(provider);
        let result = block_on(cast_value(
            Value::GpuTensor(input.clone()),
            IntegerClass::Int8,
        ));
        match mode {
            ResultMode::Failure => {
                assert!(
                    matches!(result, Err(super::super::CastError::Runtime(ref error)) if error.message().contains("injected cast execution failure"))
                );
                assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
            }
            ResultMode::Unsupported => {
                let Value::GpuTensor(output) = result.unwrap() else {
                    panic!("resident fallback")
                };
                assert_eq!(output.device_id(), input.device_id());
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&output),
                    Some(IntegerElementType::I8)
                );
                assert_eq!(provider.downloads.load(Ordering::Relaxed), 1);
                drop(output);
            }
            _ => unreachable!(),
        }
        drop(input);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn unsupported_singleton_cast_retains_exact_class_shape_owner_and_intent() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(ResultMode::Unsupported);
    let _current = ThreadProviderGuard::set(Some(provider));
    for shape in [vec![1, 1], vec![1, 1, 1, 1]] {
        for target in [
            IntegerClass::Int8,
            IntegerClass::Int16,
            IntegerClass::Int32,
            IntegerClass::Int64,
            IntegerClass::UInt8,
            IntegerClass::UInt16,
            IntegerClass::UInt32,
            IntegerClass::UInt64,
        ] {
            let mut input = provider
                .upload(&HostTensorView {
                    data: &[7.0],
                    shape: &shape,
                })
                .unwrap();
            input = input
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let Value::GpuTensor(output) =
                block_on(cast_value(Value::GpuTensor(input.clone()), target)).unwrap()
            else {
                panic!("singleton fallback must remain resident")
            };
            assert_eq!(output.device_id(), input.device_id());
            assert_eq!(output.shape(), shape);
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            let downloaded = block_on(provider.inner.download_integer(&output)).unwrap();
            assert_eq!(downloaded.shape, shape);
            use super::super::class::IntegerClassExt;
            assert_eq!(downloaded.data.element_type(), target.accelerator_type());
            assert!(block_on(provider.inner.download(&input)).is_ok());
            drop(output);
            drop(input);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn same_integer_class_cast_may_return_the_owned_input_alias() {
    let _lock = test_support::accel_test_lock();
    let provider = provider(ResultMode::Alias);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload_integer(&HostIntegerTensorView {
            data: runmat_accelerate_api::HostIntegerDataView::I8(&[3, -1]),
            shape: &[1, 2],
        })
        .unwrap();
    let Value::GpuTensor(output) = block_on(cast_value(
        Value::GpuTensor(input.clone()),
        IntegerClass::Int8,
    ))
    .unwrap() else {
        panic!("integer identity cast must remain resident");
    };
    assert!(output.shares_allocation_with(&input));
    drop(output);
    let downloaded = block_on(provider.inner.download_integer(&input)).unwrap();
    assert!(
        matches!(downloaded.data, runmat_accelerate_api::HostIntegerDataOwned::I8(values) if values == [3, -1])
    );
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    drop(input);
}
