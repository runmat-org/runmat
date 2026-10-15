use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, GpuTensorStorage, HostTensorView, NumericElementType,
};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Mode {
    Pending,
    Unsupported,
    Failed,
    Alias,
    NoAttempt,
    MissingSecond,
    FailedExpansion,
}
#[derive(Debug)]
struct DeviceFailure;
impl std::fmt::Display for DeviceFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("injected binary device failure")
    }
}
impl std::error::Error for DeviceFailure {}
struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    expanded: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "binary owning expansion fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("no host fallback in expansion test") })
    }
    fn repmat(
        &self,
        handle: &GpuTensorHandle,
        reps: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        if matches!(self.mode, Mode::NoAttempt)
            || (matches!(self.mode, Mode::MissingSecond)
                && !self.expanded.lock().unwrap().is_empty())
        {
            return Ok(None);
        }
        if matches!(self.mode, Mode::FailedExpansion) {
            return Err(
                runmat_accelerate_api::ProviderExecutionError::from_source(DeviceFailure)
                    .with_unproven_effects(),
            );
        }
        let result = (|| -> anyhow::Result<GpuTensorHandle> {
            let shape = (0..handle.shape().len().max(reps.len()))
                .map(|i| {
                    handle.shape().get(i).copied().unwrap_or(1) * reps.get(i).copied().unwrap_or(1)
                })
                .collect();
            let allocation = self.owner.allocate_zeroed(
                shape,
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.expanded.lock().unwrap().push(allocation.probe);
            Ok(allocation.handle)
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn elem_add<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.mode {
                Mode::Pending => futures::future::pending().await,
                Mode::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                    "fixture add unsupported",
                )
                .into()),
                Mode::Failed => Err(anyhow::Error::new(DeviceFailure).into()),
                Mode::Alias => Ok(left.with_view(left.view().clone())?),
                Mode::NoAttempt | Mode::MissingSecond | Mode::FailedExpansion => {
                    panic!("unavailable expansion must not invoke elementwise work")
                }
            }
        })
    }
}

#[test]
fn expanded_owned_operands_retire_on_cancel_unsupported_and_rejected_alias() {
    for mode in [
        Mode::Pending,
        Mode::Unsupported,
        Mode::Failed,
        Mode::Alias,
        Mode::NoAttempt,
        Mode::MissingSecond,
        Mode::FailedExpansion,
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            mode,
            expanded: Mutex::new(Vec::new()),
        }));
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let left = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let right = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let mut future = Box::pin(try_broadcast(
            ArithmeticProviderOperation::Add,
            provider,
            &left.handle,
            &right.handle,
        ));
        let waker = futures::task::noop_waker();
        match (mode, future.as_mut().poll(&mut Context::from_waker(&waker))) {
            (Mode::Pending, Poll::Pending) => {}
            (Mode::NoAttempt, Poll::Ready(Ok(None))) => {}
            (Mode::FailedExpansion, Poll::Ready(Err(error))) => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                let mut cause: &(dyn std::error::Error + 'static) = &error;
                let mut saw_phase = false;
                loop {
                    if let Some(provider) =
                        cause.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                    {
                        assert!(!provider.is_after_effect());
                        assert!(!provider.is_unsupported());
                        saw_phase = true;
                    }
                    if cause.is::<DeviceFailure>() {
                        assert!(saw_phase);
                        break;
                    }
                    cause = cause.source().expect("original unproven expansion cause");
                }
            }
            (Mode::Unsupported | Mode::Failed | Mode::MissingSecond, Poll::Ready(Err(error))) => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(!runmat_accelerate_api::is_unsupported_provider_error(
                    &error
                ));
                let mut cause: &(dyn std::error::Error + 'static) = &error;
                loop {
                    if let Some(provider) =
                        cause.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                    {
                        assert!(provider.is_after_effect());
                        break;
                    }
                    cause = cause.source().expect("typed after-effect cause");
                }
                if matches!(mode, Mode::Failed) {
                    let mut cause: &(dyn std::error::Error + 'static) = &error;
                    while !cause.is::<DeviceFailure>() {
                        cause = cause
                            .source()
                            .expect("original binary device failure retained");
                    }
                }
            }
            (Mode::Alias, Poll::Ready(Err(error))) => {
                assert!(error.message().contains("invalid element-wise output"))
            }
            (_, result) => panic!("unexpected production expansion result: {result:?}"),
        }
        drop(future);
        let expanded = provider.expanded.lock().unwrap();
        assert_eq!(
            expanded.len(),
            match mode {
                Mode::NoAttempt | Mode::FailedExpansion => 0,
                Mode::MissingSecond => 1,
                _ => 2,
            }
        );
        assert!(expanded.iter().all(|probe| probe.retirements() == 1));
        assert_eq!(left.probe.retirements(), 0);
        assert_eq!(right.probe.retirements(), 0);
        left.handle.allocation_lease().validate_live().unwrap();
        right.handle.allocation_lease().validate_live().unwrap();
    }
}
