//! The host-A/resident-H path admits its native readback before any effect.
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
fn resident_double_h_host_uint8_a_preserves_class_and_pre_effect_budget() {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let kernel = Tensor::new(vec![2.0], vec![1, 1]).expect("double H");
    let resident_h = gpu_helpers::upload_tensor(provider, &kernel).expect("resident H");
    let image = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U8)
        .expect("uint8 A");
    let args = [Value::Tensor(image), Value::GpuTensor(resident_h.clone())];

    provider.reset_telemetry();
    let session = Rc::new(PlacementSession::new(Default::default()));
    let result = block_on(call_builtin_async_in_context(
        &context(session.clone(), 1_000_000),
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
        session
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
        &context(Rc::new(PlacementSession::new(Default::default())), 1),
        "imfilter",
        &args,
    ))
    .expect_err("finite quota must reject before H readback");
    assert!(rejected
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(resident_h.allocation_lease().validate_live().is_ok());
}
