use super::super::binary;
use crate::builtins::common::test_support;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_builtins::LogicalBinaryOperator;
use runmat_value::{Tensor, Value};
use std::rc::Rc;
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum HookBehavior {
    Unsupported,
    Failure,
    Malformed,
}

struct LogicalHookProvider {
    inner: InProcessProvider,
    behavior: HookBehavior,
    malformed_output: Mutex<Option<runmat_accelerate_api::allocation::GpuTensorWireDescriptor>>,
}

impl LogicalHookProvider {
    fn new(behavior: HookBehavior) -> Self {
        Self {
            inner: InProcessProvider::new(),
            behavior,
            malformed_output: Mutex::new(None),
        }
    }

    fn invoke(&self) -> anyhow::Result<GpuTensorHandle> {
        match self.behavior {
            HookBehavior::Unsupported => {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "logical hook is unavailable",
                ))
            }
            HookBehavior::Failure => anyhow::bail!("injected logical kernel failure"),
            HookBehavior::Malformed => {
                let output = self.inner.upload(&HostTensorView {
                    data: &[1.0, 0.0],
                    shape: &[2, 1],
                })?;
                *self.malformed_output.lock().expect("malformed output lock") =
                    Some(output.wire_descriptor());
                Ok(output)
            }
        }
    }
}

impl AccelProvider for LogicalHookProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn upload_logical_raw(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_logical_raw(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn logical_and(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.invoke()
    }
}

fn provider(behavior: HookBehavior) -> &'static LogicalHookProvider {
    Box::leak(Box::new(LogicalHookProvider::new(behavior)))
}

fn resident_pair(provider: &dyn AccelProvider) -> (GpuTensorHandle, GpuTensorHandle) {
    let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor");
    let data = tensor.materialize_f64();
    let view = HostTensorView {
        data: &data,
        shape: tensor.shape(),
    };
    (
        provider.upload(&view).expect("upload lhs"),
        provider.upload(&view).expect("upload rhs"),
    )
}

#[test]
fn typed_unsupported_hook_uses_exact_host_fallback() {
    let _state = test_support::accel_test_lock();
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let provider = provider(HookBehavior::Unsupported);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let (lhs, rhs) = resident_pair(provider);

    let result = block_on(binary::execute(
        Some(&context),
        Value::GpuTensor(lhs.clone()),
        Value::GpuTensor(rhs.clone()),
        LogicalBinaryOperator::And,
    ))
    .expect("typed unsupported hook should fall back");
    let gathered = test_support::gather(result.value().clone()).expect("logical result");
    assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0]);
    lhs.release().expect("free lhs");
    rhs.release().expect("free rhs");
}

#[test]
fn provider_failure_remains_visible() {
    let _state = test_support::accel_test_lock();
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let provider = provider(HookBehavior::Failure);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let (lhs, rhs) = resident_pair(provider);

    let error = block_on(binary::execute(
        Some(&context),
        Value::GpuTensor(lhs.clone()),
        Value::GpuTensor(rhs.clone()),
        LogicalBinaryOperator::And,
    ))
    .expect_err("provider failure must remain visible");
    assert_eq!(
        error.identifier(),
        Some("RunMat:gpu:ProviderExecutionFailed")
    );
    assert!(error.message().contains("injected logical kernel failure"));
    lhs.release().expect("free lhs");
    rhs.release().expect("free rhs");
}

#[test]
fn malformed_provider_output_is_freed_and_rejected() {
    let _state = test_support::accel_test_lock();
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let provider = provider(HookBehavior::Malformed);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let (lhs, rhs) = resident_pair(provider);

    let error = block_on(binary::execute(
        Some(&context),
        Value::GpuTensor(lhs.clone()),
        Value::GpuTensor(rhs.clone()),
        LogicalBinaryOperator::And,
    ))
    .expect_err("malformed provider output must be rejected");
    assert_eq!(
        error.identifier(),
        Some("RunMat:gpu:ProviderPayloadMismatch")
    );
    let malformed = provider
        .malformed_output
        .lock()
        .expect("malformed output lock")
        .take()
        .expect("provider recorded malformed output");
    assert!(
        provider.inner.acquire_live(&malformed).is_err(),
        "rejected output has no surviving owned reference"
    );
    lhs.release().expect("free lhs");
    rhs.release().expect("free rhs");
}
