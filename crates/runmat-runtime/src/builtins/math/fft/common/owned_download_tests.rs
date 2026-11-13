use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, HostTensorView, NumericElementType,
};
use std::{
    future::Future,
    task::{Context, Poll},
};

enum Mode {
    Success,
    Error,
    Pending,
}
struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "FFT owned readback fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            match self.mode {
                Mode::Success => Ok(HostTensorOwned {
                    data: vec![0.0, 0.0],
                    shape: handle.shape().to_vec(),
                    storage: GpuTensorStorage::Real,
                }),
                Mode::Error => anyhow::bail!("fixture readback failed"),
                Mode::Pending => futures::future::pending().await,
            }
        })
    }
}

#[test]
fn owning_readback_releases_success_and_error_intermediates() {
    for mode in [Mode::Success, Mode::Error] {
        let provider = Provider {
            owner: TestAllocationOwner::new(),
            mode,
        };
        let allocation = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let result = futures::executor::block_on(download_owned_provider_complex_tensor(
            &provider,
            allocation.handle,
            "ifft",
        ));
        match provider.mode {
            Mode::Success => assert_eq!(
                result.unwrap().materialize_f64(),
                vec![(0.0, 0.0), (0.0, 0.0)]
            ),
            Mode::Error => assert!(result
                .unwrap_err()
                .message()
                .contains("fixture readback failed")),
            Mode::Pending => unreachable!(),
        }
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[test]
fn abandoning_owned_readback_preserves_independent_borrowed_alias() {
    let provider = Provider {
        owner: TestAllocationOwner::new(),
        mode: Mode::Pending,
    };
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut future = Box::pin(download_owned_provider_complex_tensor(
        &provider,
        allocation.handle.clone(),
        "ifft",
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    drop(future);
    assert_eq!(allocation.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn abandoning_sole_owned_readback_releases_the_pending_intermediate() {
    let provider = Provider {
        owner: TestAllocationOwner::new(),
        mode: Mode::Pending,
    };
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut future = Box::pin(download_owned_provider_complex_tensor(
        &provider,
        allocation.handle,
        "ifft",
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    drop(future);
    assert_eq!(allocation.probe.retirements(), 1);
}
