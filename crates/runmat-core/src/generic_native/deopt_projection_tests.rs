//! Native-to-VM completion projection with actual owned provider allocations.
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, ApiDeviceInfo, GpuTensorHandle,
    HostTensorView, ThreadProviderGuard,
};
use runmat_native_executor::deopt::{DeoptimizationPolicy, FaultInjection, ResumeTarget};
use runmat_value::Value;
use std::sync::{LazyLock, Mutex};

struct TrackingProvider {
    inner: InProcessProvider,
    outputs: Mutex<Vec<GpuTensorHandle>>,
    freed: Mutex<Vec<u64>>,
}
impl AccelProvider for TrackingProvider {
    fn upload(&self, value: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(value)
    }
    fn download<'a>(&'a self, value: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(value)
    }
    fn free(&self, value: &GpuTensorHandle) -> anyhow::Result<()> {
        self.inner.free(value)?;
        self.freed.lock().unwrap().push(value.buffer_id);
        Ok(())
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
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let output = self.inner.unary_sin(input).await?;
            self.outputs.lock().unwrap().push(output.clone());
            Ok(output)
        })
    }
}

static PROVIDER: LazyLock<TrackingProvider> = LazyLock::new(|| TrackingProvider {
    inner: InProcessProvider::new(),
    outputs: Mutex::new(Vec::new()),
    freed: Mutex::new(Vec::new()),
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
            let result = block_on(super::invoke_with_policy(
                &unit,
                super::DeoptimizationInvocation {
                    executor: compiled.executor,
                    function: compiled.entrypoint,
                    captures: Vec::new(),
                    arguments: vec![Value::GpuTensor(input.clone())],
                    requested_outputs,
                    runtime: session.runtime_context().clone(),
                    policy: DeoptimizationPolicy {
                        target: ResumeTarget::Interpreter,
                        ..Default::default()
                    }
                    .inject(FaultInjection::Safepoint(safepoint)),
                    osr: None,
                    workspace: None,
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
                    assert_eq!(value.buffer_id, input.buffer_id);
                }
            }
            let discarded = PROVIDER.outputs.lock().unwrap()[first..].to_vec();
            assert!(
                !discarded.is_empty(),
                "resumed body must execute the actual sine provider hook"
            );
            for value in discarded {
                assert!(block_on(PROVIDER.download(&value)).is_err());
                assert_eq!(
                    PROVIDER
                        .freed
                        .lock()
                        .unwrap()
                        .iter()
                        .filter(|id| **id == value.buffer_id)
                        .count(),
                    1
                );
            }
            assert_eq!(
                block_on(PROVIDER.download(&input)).unwrap().data,
                vec![0.25, 0.5]
            );
            runmat_gc::gc_unregister_root(caller).unwrap();
            PROVIDER.free(&input).unwrap();
        }
    });
}
