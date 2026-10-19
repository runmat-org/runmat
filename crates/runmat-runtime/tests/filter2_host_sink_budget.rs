//! Canonical dispatch uses one runtime library for filter2's host-result
//! resident readback route and its shared placement service.

use futures::executor::block_on;
use runmat_accelerate::{placement::PlacementSession, simple_provider::InProcessProvider};
use runmat_accelerate_api::{AccelProvider, ThreadProviderGuard};
use runmat_runtime::{
    builtins::common::gpu_helpers,
    call_builtin_async_in_context,
    context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    },
};
use runmat_value::{NumericDType, Tensor, Value};
use std::rc::Rc;

struct HostBudget(u64);
impl RuntimeParallelService for HostBudget {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(self.0),
            epoch: 0,
        }
    }
}

fn context(session: Rc<PlacementSession>, host_bytes: u64) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session)
            .with_parallel(Rc::new(HostBudget(host_bytes))),
    )
}

#[test]
fn resident_h_host_image_is_one_admitted_readback_and_no_effect_below_budget() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let kernel = Tensor::new(vec![2.0], vec![1, 1]).expect("H");
    let resident_h = gpu_helpers::upload_tensor(provider, &kernel).expect("resident H");
    let image = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("X");
    let args = [Value::GpuTensor(resident_h.clone()), Value::Tensor(image)];

    provider.reset_telemetry();
    let session = Rc::new(PlacementSession::new(Default::default()));
    let result = block_on(call_builtin_async_in_context(
        &context(session.clone(), 1_000_000),
        "filter2",
        &args,
    ))
    .expect("admitted readback");
    let Value::Tensor(result) = result.value() else {
        panic!("non-scalar host result must remain host tensor")
    };
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
    assert!(provider.telemetry_snapshot().download_bytes > 0);
    assert_eq!(
        session
            .profile_snapshot()
            .feedback
            .iter()
            .filter(|sample| sample.signature.operation == "filter2")
            .map(|sample| sample.samples)
            .sum::<u64>(),
        1
    );

    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(
        &context(Rc::new(PlacementSession::new(Default::default())), 1),
        "filter2",
        &args,
    ))
    .expect_err("finite host quota must reject before H readback");
    assert!(error
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(resident_h.allocation_lease().validate_live().is_ok());

    provider.reset_telemetry();
    let invalid = block_on(call_builtin_async_in_context(
        &context(
            Rc::new(PlacementSession::new(Default::default())),
            1_000_000,
        ),
        "filter2",
        &[
            Value::GpuTensor(resident_h.clone()),
            Value::from("not-an-image"),
        ],
    ))
    .expect_err("invalid host input must fail before readback");
    assert_eq!(invalid.identifier(), Some("RunMat:filter2:InvalidInput"));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
}

#[test]
fn integer_host_h_with_resident_x_uses_admitted_host_route_without_upload() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let image = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("X");
    let resident_x = gpu_helpers::upload_tensor(provider, &image).expect("resident X");
    let kernel =
        Tensor::new_with_dtype(vec![2.0], vec![1, 1], NumericDType::U8).expect("integer H");
    let args = [Value::Tensor(kernel), Value::GpuTensor(resident_x.clone())];

    provider.reset_telemetry();
    let session = Rc::new(PlacementSession::new(Default::default()));
    let result = block_on(call_builtin_async_in_context(
        &context(session.clone(), 1_000_000),
        "filter2",
        &args,
    ))
    .expect("admitted host evaluator");
    let Value::Tensor(result) = result.value() else {
        panic!("host result expected")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F64);
    assert_eq!(result.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
    let telemetry = provider.telemetry_snapshot();
    assert!(telemetry.download_bytes > 0);
    assert_eq!(telemetry.upload_bytes, 0);
    assert_eq!(
        session
            .profile_snapshot()
            .feedback
            .iter()
            .filter(|sample| sample.signature.operation == "filter2")
            .map(|sample| sample.samples)
            .sum::<u64>(),
        1
    );

    provider.reset_telemetry();
    let rejected = block_on(call_builtin_async_in_context(
        &context(Rc::new(PlacementSession::new(Default::default())), 1),
        "filter2",
        &args,
    ))
    .expect_err("finite quota rejects before X readback");
    assert!(rejected
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    let telemetry = provider.telemetry_snapshot();
    assert_eq!(telemetry.download_bytes, 0);
    assert_eq!(telemetry.upload_bytes, 0);
    assert!(resident_x.allocation_lease().validate_live().is_ok());
}
