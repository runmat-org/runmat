//! The host-A/resident-H path admits its native readback before any effect.
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{AccelProvider, ThreadProviderGuard};
use runmat_runtime::{builtins::common::gpu_helpers, call_builtin_async_in_context};
use runmat_value::{NumericDType, Tensor, Value};
#[path = "support/budgeted_placement.rs"]
mod budgeted_placement;
use budgeted_placement::{assert_capacity_denied, context, session};

#[test]
fn resident_double_h_host_uint8_a_preserves_class_and_pre_effect_budget() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let kernel = Tensor::new(vec![2.0], vec![1, 1]).expect("double H");
    let resident_h = gpu_helpers::upload_tensor(provider, &kernel).expect("resident H");
    let image = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U8)
        .expect("uint8 A");
    let args = [Value::Tensor(image), Value::GpuTensor(resident_h.clone())];

    provider.reset_telemetry();
    let placement = session();
    let result = block_on(call_builtin_async_in_context(
        &context(placement.clone(), 1_000_000),
        "imfilter",
        &args,
    ))
    .expect("admitted host result");
    let Value::Tensor(result) = result.value() else {
        panic!("multi-element uint8 output must remain a host tensor")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::U8);
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
    assert!(provider.telemetry_snapshot().download_bytes > 0);
    assert_eq!(
        placement
            .profile_snapshot()
            .feedback
            .iter()
            .filter(|sample| sample.signature.operation == "imfilter")
            .map(|sample| sample.samples)
            .sum::<u64>(),
        1
    );

    provider.reset_telemetry();
    let rejected = block_on(call_builtin_async_in_context(
        &context(session(), 1),
        "imfilter",
        &args,
    ))
    .expect_err("finite quota must reject before H readback");
    assert_capacity_denied(&rejected);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(resident_h.allocation_lease().validate_live().is_ok());
}
