use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProviderFuture, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostTensorView, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Mode {
    Success,
    UnsupportedCast,
    FailedMultiply,
    PendingCast,
    MalformedCast,
}

struct Provider {
    owner: TestAllocationOwner,
    records: Mutex<Vec<(u64, RetirementProbe)>>,
    mode: Mode,
}
impl Provider {
    fn numeric(&self, data: Vec<f64>) -> GpuTensorHandle {
        let allocation = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(data),
                shape: vec![2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.buffer_id(), allocation.probe));
        allocation.handle
    }
    fn data(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostNumericDataOwned> {
        let probe = self
            .records
            .lock()
            .unwrap()
            .iter()
            .find(|(id, _)| *id == handle.buffer_id())
            .unwrap()
            .1
            .clone();
        probe.with_data(handle, |data| match data {
            HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
            HostNumericDataView::U8(data) => HostNumericDataOwned::U8(data.to_vec()),
            _ => panic!("fixture class"),
        })
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owning extrema index stages".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("typed download required") })
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostNumericTensorOwned {
                data: self.data(input)?,
                shape: input.shape().to_vec(),
                storage: input.physical().storage(),
            })
        })
    }
    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.mode {
                Mode::UnsupportedCast => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "no numeric cast",
                    ))
                }
                Mode::MalformedCast => return Ok(input.clone()),
                _ => (),
            }
            let HostNumericDataOwned::U8(data) = self.data(input)? else {
                panic!("expected logical storage")
            };
            let output = self.numeric(data.into_iter().map(f64::from).collect());
            if matches!(self.mode, Mode::PendingCast) {
                std::future::pending::<()>().await;
            }
            Ok(output)
        })
    }
    fn scalar_mul(&self, input: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        assert!(
            !input.view().is_logical(),
            "logical views cannot be arithmetic inputs"
        );
        if matches!(self.mode, Mode::FailedMultiply) {
            return Err(
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "multiply failed").into(),
            );
        }
        let HostNumericDataOwned::F64(data) = self.data(input)? else {
            panic!("expected Double")
        };
        Ok(self.numeric(data.into_iter().map(|value| value * scalar).collect()))
    }
    fn scalar_add(&self, input: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        assert!(!input.view().is_logical());
        let HostNumericDataOwned::F64(data) = self.data(input)? else {
            panic!("expected Double")
        };
        Ok(self.numeric(data.into_iter().map(|value| value + scalar).collect()))
    }
}

#[test]
fn index_stages_preserve_class_aliases_and_retire_intermediates_on_all_exits() {
    for mode in [
        Mode::Success,
        Mode::UnsupportedCast,
        Mode::FailedMultiply,
        Mode::PendingCast,
        Mode::MalformedCast,
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            records: Mutex::new(Vec::new()),
            mode,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_logical(vec![1, 0], vec![2, 1])
            .unwrap();
        provider
            .records
            .lock()
            .unwrap()
            .push((input.handle.buffer_id(), input.probe.clone()));
        let mut future = Box::pin(ExtremaKind::Max.mask_indices(provider, input.handle.clone()));
        if matches!(mode, Mode::PendingCast) {
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            assert_eq!(provider.records.lock().unwrap()[1].1.retirements(), 0);
            drop(future);
        } else {
            let output = futures::executor::block_on(future);
            match mode {
                Mode::Success => {
                    let Value::GpuTensor(output) = output.unwrap() else {
                        panic!("resident Double indices")
                    };
                    assert_eq!(
                        output.physical().element_type(),
                        runmat_accelerate_api::NumericElementType::F64
                    );
                    assert_eq!(
                        provider.data(&output).unwrap(),
                        HostNumericDataOwned::F64(vec![1.0, 2.0])
                    );
                    drop(output);
                }
                Mode::UnsupportedCast => assert_eq!(
                    output.unwrap(),
                    Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap())
                ),
                Mode::FailedMultiply | Mode::MalformedCast => {
                    let error = output.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    if matches!(mode, Mode::FailedMultiply) {
                        let mut source = std::error::Error::source(&error);
                        let mut found = false;
                        while let Some(error) = source {
                            found |= error.downcast_ref::<std::io::Error>().is_some_and(|error| {
                                error.kind() == std::io::ErrorKind::BrokenPipe
                            });
                            source = error.source();
                        }
                        assert!(found, "original failure must remain downcastable");
                    }
                }
                Mode::PendingCast => unreachable!(),
            }
        }
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        for (_, probe) in provider.records.lock().unwrap().iter().skip(1) {
            assert_eq!(probe.retirements(), 1);
        }
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
