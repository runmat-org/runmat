//! Native-to-VM completion projection with actual owned provider allocations.
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::allocation::WeakGpuTensorHandle;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, ApiDeviceInfo, GpuTensorHandle,
    HostTensorView, ThreadProviderGuard,
};
use runmat_native_executor::deopt::{DeoptimizationPolicy, FaultInjection, ResumeTarget};
use runmat_value::Value;
use std::sync::{LazyLock, Mutex};

struct TrackingProvider {
    inner: InProcessProvider,
    outputs: Mutex<
        Vec<(
            WeakGpuTensorHandle,
            runmat_accelerate_api::allocation::GpuTensorWireDescriptor,
        )>,
    >,
}
impl AccelProvider for TrackingProvider {
    fn upload(&self, value: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(value)
    }
    fn download<'a>(&'a self, value: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(value)
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn device_info_struct(&self) -> ApiDeviceInfo {
        self.inner.device_info_struct()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn capability_snapshot(&self) -> runmat_accelerate_api::ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }
    fn capability_snapshot_ref(
        &self,
    ) -> Option<&runmat_accelerate_api::ProviderCapabilitySnapshot> {
        self.inner.capability_snapshot_ref()
    }
    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        self.inner.placement_resources()
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.inner.query_feasibility(query)
    }
    fn estimate_cost(
        &self,
        query: &runmat_accelerate_api::ProviderCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        self.inner.estimate_cost(query)
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let output = self.inner.unary_sin(input).await?;
            self.outputs
                .lock()
                .unwrap()
                .push((output.downgrade(), output.wire_descriptor()));
            Ok(output)
        })
    }
}

static PROVIDER: LazyLock<TrackingProvider> = LazyLock::new(|| TrackingProvider {
    inner: InProcessProvider::new(),
    outputs: Mutex::new(Vec::new()),
});
static LOCK: Mutex<()> = Mutex::new(());

#[test]
fn native_deopt_projection_releases_discarded_slots_on_success_and_output_error() {
    runmat_gc::gc_test_context(|| {
        let _lock = LOCK.lock().unwrap();
        let _provider = ThreadProviderGuard::set(Some(&*PROVIDER));
        for (source, requested_outputs, fails) in [
            ("function y = projectedNative(x)\ndiscarded = sin(x);\ny = x;\nend\n", 1, false),
            ("function varargout = projectedNative(x)\ndiscarded = sin(x);\nvarargout = 3;\nend\n", 1, true),
            ("function [y, varargout] = projectedNative(x)\ndiscarded = sin(x);\ny = x;\nvarargout = 3;\nend\n", 2, true),
            ("function [y, varargout] = projectedNative(x)\ndiscarded = sin(x);\ny = x;\nvarargout = {x};\nend\n", 2, false),
        ] {
            let mut session = crate::RunMatSession::with_options(false, false).unwrap();
            session.configure_runtime_context();
            let unit = block_on(session.compile_executable_unit(
                crate::ExecutableSource::new("deopt-projection@1", "projectedNative.m", source),
                None,
            ))
            .unwrap();
            let compiled = super::super::compile::compile(&unit, Some("projectedNative")).unwrap();
            let safepoint = compiled
                .safepoints
                .get(&compiled.entrypoint)
                .and_then(|points| points.first())
                .copied()
                .expect("real call safepoint");
            let input = PROVIDER
                .upload(&HostTensorView {
                    data: &[0.25, 0.5],
                    shape: &[1, 2],
                })
                .unwrap();
            let caller = runmat_gc::gc_register_value_root(Box::new(runmat_gc::GlobalRoot::new(
                vec![Value::GpuTensor(input.clone())],
                "independent deopt caller".into(),
            )))
            .unwrap();
            let first = PROVIDER.outputs.lock().unwrap().len();
            let runtime = unit
                .bind_runtime_sources(session.runtime_context().clone())
                .expect("bind exact projected native source for VM resume");
            let result = block_on(super::invoke_with_policy(
                &unit,
                super::DeoptimizationInvocation {
                    executor: compiled.executor,
                    function: compiled.entrypoint,
                    lexical_environment: None,
                    inputs: super::Inputs::Source { captures: Vec::new(), arguments: vec![Value::GpuTensor(input.clone())] },
                    requested_outputs,
                    runtime,
                    policy: DeoptimizationPolicy {
                        target: ResumeTarget::Interpreter,
                        ..Default::default()
                    }
                    .inject(FaultInjection::Safepoint(safepoint)),
                    osr: None,
                    workspace: None,
                    dynamic_admission: None,
                },
            ));
            if fails {
                assert!(result.unwrap_err().to_string().contains("varargout"));
            } else {
                let output = result.unwrap();
                assert_eq!(output.outputs.len(), requested_outputs);
                for value in &output.outputs {
                    let Value::GpuTensor(value) = value else {
                        panic!("native-deopt resident alias")
                    };
                    assert!(value.shares_allocation_with(&input));
                }
            }
            let discarded = PROVIDER.outputs.lock().unwrap()[first..].to_vec();
            assert!(
                !discarded.is_empty(),
                "resumed body must execute the actual sine provider hook"
            );
            for (weak, descriptor) in discarded {
                assert!(weak.acquire().is_err(), "discarded view must be released");
                assert!(PROVIDER.inner.acquire_live(&descriptor).is_err(), "retired allocation must not be reacquired");
            }
            assert_eq!(
                block_on(PROVIDER.download(&input)).unwrap().data,
                vec![0.25, 0.5]
            );
            runmat_gc::gc_unregister_root(caller).unwrap();
            drop(input);
        }
    });
}
