use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProviderFuture, GpuTensorStorage, HostTensorView, NumericElementType,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Fault {
    Pending,
    Failure,
    WrongClass,
}
struct Provider {
    owner: TestAllocationOwner,
    fault: Fault,
    calls: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
}

#[derive(Debug)]
struct BinaryFailure;
impl std::fmt::Display for BinaryFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("injected binary device failure")
    }
}
impl std::error::Error for BinaryFailure {}

impl Provider {
    fn binary_failure(
        &self,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'_, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::Error::new(BinaryFailure).into())
        })
    }
}

impl AccelProvider for Provider {
    fn elem_add<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.binary_failure()
    }
    fn elem_sub<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.binary_failure()
    }
    fn elem_mul<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.binary_failure()
    }
    fn elem_div<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.binary_failure()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "arrayfun owning lifecycle fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected upload after admitted resident sin")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("operational failure must not retry with download")
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.fault {
                Fault::Pending => futures::future::pending().await,
                Fault::Failure => anyhow::bail!("injected arrayfun sine device failure"),
                Fault::WrongClass => {
                    let output = self.owner.allocate_zeroed(
                        input.shape().to_vec(),
                        NumericElementType::I64,
                        GpuTensorStorage::Real,
                    )?;
                    self.outputs.lock().unwrap().push(output.probe);
                    Ok(output.handle)
                }
            }
        })
    }
}

#[test]
fn automatic_binary_operational_errors_keep_source_and_never_retry() {
    runmat_gc::gc_test_context(|| {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            fault: Fault::Failure,
            calls: AtomicUsize::new(0),
            outputs: Mutex::new(Vec::new()),
        }));
        let _scope = ThreadProviderGuard::set(Some(provider));
        for name in ["plus", "minus", "times", "rdivide", "ldivide"] {
            let left = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let right = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let left_handle = left
                .handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .unwrap();
            let right_handle = right
                .handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .unwrap();
            let before = provider.calls.load(Ordering::SeqCst);
            let error = mapped(
                name,
                vec![
                    Value::GpuTensor(left_handle),
                    Value::GpuTensor(right_handle),
                ],
            )
            .unwrap_err();
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                before + 1,
                "{name}: one actual provider hook"
            );
            let mut cursor: Option<&(dyn std::error::Error + 'static)> = Some(&error);
            let mut retained = false;
            let mut terminal = false;
            while let Some(cause) = cursor {
                retained |= cause.is::<BinaryFailure>();
                terminal |= cause
                    .downcast_ref::<runmat_runtime::RuntimeError>()
                    .is_some_and(|error| {
                        error.gpu_gather_retry == runmat_runtime::GpuGatherRetry::Never
                    });
                cursor = cause.source();
            }
            assert!(
                retained && terminal,
                "{name}: original typed cause and terminal retry policy: {error:?}"
            );
            assert_eq!(left.probe.retirements(), 0);
            assert_eq!(right.probe.retirements(), 0);
            drop(left.handle);
            drop(right.handle);
            assert_eq!(left.probe.retirements(), 1);
            assert_eq!(right.probe.retirements(), 1);
        }
    });
}

#[test]
fn pending_failed_and_invalid_batch_keep_true_lease_lifetimes() {
    runmat_gc::gc_test_context(|| {
        let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name("sin");
        assert_eq!(original.len(), 1);
        assert!(
            runmat_runtime::builtins::common::spec::selected_builtin_binding(
                "sin",
                Some(original[0].identity),
                None
            )
            .is_some()
        );
        for fault in [Fault::Pending, Fault::Failure, Fault::WrongClass] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                fault,
                calls: AtomicUsize::new(0),
                outputs: Mutex::new(Vec::new()),
            }));
            let _scope = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let input_probe = input.probe;
            let mut future = Box::pin(async move {
                let args = [
                    Value::FunctionHandle("sin".into()),
                    Value::GpuTensor(input.handle),
                ];
                call_builtin_async_with_outputs("arrayfun", &args, 1).await
            });
            let waker = futures::task::noop_waker();
            let outcome = future.as_mut().poll(&mut Context::from_waker(&waker));
            match fault {
                Fault::Pending => {
                    assert!(outcome.is_pending());
                    assert_eq!(input_probe.retirements(), 0);
                }
                Fault::Failure | Fault::WrongClass => {
                    let Poll::Ready(Err(error)) = outcome else {
                        panic!("admitted provider failure must be retained")
                    };
                    assert!(error.message().starts_with("arrayfun:"), "{error:?}");
                    assert!(std::error::Error::source(&error).is_some());
                }
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            drop(future);
            assert_eq!(input_probe.retirements(), 1);
            for output in provider.outputs.lock().unwrap().iter() {
                assert_eq!(
                    output.retirements(),
                    1,
                    "invalid owned output retires exactly once"
                );
            }
        }
    });
}
