//! Exercise canonical `imhist` dispatch with one runtime library and the real
//! placement service. A unit-test crate would duplicate the runtime trait that
//! `runmat-accelerate` implements.
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{AccelProvider, ThreadProviderGuard};
use runmat_runtime::{builtins::common::gpu_helpers, call_builtin_async_with_outputs_in_context};
use runmat_value::{NumericDType, Tensor, Value};
#[path = "support/budgeted_placement.rs"]
mod budgeted_placement;
use budgeted_placement::{assert_capacity_denied, context, session};

#[test]
fn finite_host_budget_admits_bound_and_rejects_before_gather() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = Tensor::new_with_dtype(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2], NumericDType::F64)
        .expect("histogram input");
    let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
    let args = [Value::GpuTensor(source.clone()), Value::Num(3.0)];

    provider.reset_telemetry();
    let admitted = block_on(call_builtin_async_with_outputs_in_context(
        &context(session(), 1_000_000),
        "imhist",
        &args,
        1,
    ))
    .expect("finite host budget admits real route");
    let Value::OutputList(outputs) = &*admitted else {
        panic!("one requested output must retain its result");
    };
    assert!(matches!(outputs.as_slice(), [Value::GpuTensor(_)]));
    assert!(provider.telemetry_snapshot().download_bytes > 0);

    provider.reset_telemetry();
    let rejected = block_on(call_builtin_async_with_outputs_in_context(
        &context(session(), 1),
        "imhist",
        &args,
        1,
    ))
    .expect_err("finite budget must reject selected route before gather");
    assert_capacity_denied(&rejected);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(source.allocation_lease().validate_live().is_ok());
}
