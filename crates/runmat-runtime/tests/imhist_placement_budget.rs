//! Exercise canonical `imhist` dispatch with one runtime library and the real
//! placement service. A unit-test crate would duplicate the runtime trait that
//! `runmat-accelerate` implements.
use futures::executor::block_on;
use runmat_accelerate::{placement::PlacementSession, simple_provider::InProcessProvider};
use runmat_accelerate_api::{AccelProvider, ThreadProviderGuard};
use runmat_runtime::{
    builtins::common::gpu_helpers,
    call_builtin_async_with_outputs_in_context,
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

fn budgeted_context(host_bytes: u64) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(Rc::new(PlacementSession::new(Default::default())))
            .with_parallel(Rc::new(HostBudget(host_bytes))),
    )
}

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
        &budgeted_context(1024),
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
        &budgeted_context(1),
        "imhist",
        &args,
        1,
    ))
    .expect_err("finite budget must reject selected route before gather");
    assert!(rejected
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(source.allocation_lease().validate_live().is_ok());
}
