use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, NumericElementType,
    ThreadProviderGuard,
};
use runmat_value::Tensor;
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Mode {
    Pending,
    Failure,
    Unsupported,
    AliasedOutput,
}

struct Provider {
    owner: TestAllocationOwner,
    uploads: Mutex<Vec<RetirementProbe>>,
    mode: Mode,
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "interp1 owning lifecycle fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_zeroed(
            host.shape.to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.uploads.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected host transfer") })
    }
    fn interp1<'a>(
        &'a self,
        request: &'a ProviderInterp1Request<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.mode {
                Mode::Pending => futures::future::pending().await,
                Mode::Failure => anyhow::bail!("fixture interpolation failure"),
                Mode::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                    "interp1 not supported by provider",
                )),
                Mode::AliasedOutput => Ok(request.xq.with_view(request.xq.view().clone())?),
            }
        })
    }
}

#[test]
fn invalid_explicit_x_is_rejected_before_resident_y_download() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        uploads: Mutex::new(Vec::new()),
        mode: Mode::Failure,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident Y fixture");
    let result = futures::executor::block_on(interp1_builtin(vec![
        Value::Tensor(Tensor::new(vec![2.0, 1.0, 2.0], vec![1, 3]).unwrap()),
        Value::GpuTensor(input.handle.clone()),
        Value::Num(1.5),
    ]));
    let error = result.expect_err("duplicate X must be rejected before the download hook");
    assert!(error.message().contains("distinct"), "{error}");
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn interpolation_uploads_retire_on_pending_drop_failure_unsupported_and_alias_rejection() {
    for mode in [
        Mode::Pending,
        Mode::Failure,
        Mode::Unsupported,
        Mode::AliasedOutput,
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            uploads: Mutex::new(Vec::new()),
            mode,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let args = [
            Value::GpuTensor(input.handle.clone()),
            Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![2, 1]).unwrap()),
        ];
        let mut future = Box::pin(try_interp1_gpu(
            crate::call::FunctionArgumentView::from_slice(&args),
            None,
        ));
        let waker = futures::task::noop_waker();
        match (mode, future.as_mut().poll(&mut Context::from_waker(&waker))) {
            (Mode::Pending, Poll::Pending) | (Mode::Unsupported, Poll::Ready(Ok(None))) => {}
            (Mode::Failure, Poll::Ready(Err(error))) => assert!(error.source.is_some()),
            (Mode::AliasedOutput, Poll::Ready(Err(error))) => {
                assert!(error.message().contains("invalid interpolation result"))
            }
            (_, actual) => panic!("unexpected interpolation lifecycle outcome: {actual:?}"),
        }
        drop(future);
        let uploads = provider.uploads.lock().unwrap();
        assert_eq!(uploads.len(), 2);
        assert!(uploads.iter().all(|probe| probe.retirements() == 1));
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
        drop(args);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
