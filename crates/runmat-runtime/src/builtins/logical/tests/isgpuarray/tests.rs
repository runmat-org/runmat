use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_accelerate_api::GpuHandleProvenance;
use runmat_value::Tensor;

fn run(value: Value) -> bool {
    match block_on(isgpuarray_builtin(value)).expect("isgpuarray") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn only_explicit_handles_report_true() {
    assert!(!run(Value::Num(1.0)));
    test_support::with_test_provider(|provider| {
        let handle =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0], vec![1, 1]).unwrap())
                .unwrap();
        assert!(!run(Value::GpuTensor(handle.clone())));
        assert!(run(Value::GpuTensor(
            handle
                .clone()
                .with_provenance(GpuHandleProvenance::Explicit)
        )));
        provider.free(&handle).ok();
    });
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(isgpuarray_builtin(Value::Num(1.0))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:isgpuarray:TooManyOutputs"));
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_explicit_identity_is_distinct_from_automatic_residency() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle =
        gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0], vec![1, 1]).unwrap()).unwrap();
    assert!(!run(Value::GpuTensor(handle.clone())));
    assert!(run(Value::GpuTensor(
        handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
    )));
    provider.free(&handle).ok();
}
