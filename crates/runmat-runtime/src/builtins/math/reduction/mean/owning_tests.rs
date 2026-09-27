use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProviderFuture, GpuTensorStorage, HostTensorView, NumericElementType,
};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

struct Provider {
    owner: TestAllocationOwner,
    permutations: Mutex<Vec<RetirementProbe>>,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Pending,
    Unsupported,
    Failure,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "mean pending reshape ownership".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn permute(&self, input: &GpuTensorHandle, order: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let shape = order.iter().map(|&axis| input.shape()[axis]).collect();
        let allocation =
            self.owner
                .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)?;
        self.permutations.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn reshape(&self, input: &GpuTensorHandle, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Ok(input.with_view(input.view().clone().with_shape(shape.to_vec()))?)
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.mode {
                Mode::Pending => futures::future::pending().await,
                Mode::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                    "mean dimension",
                )),
                Mode::Failure => Err(anyhow::Error::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "mean provider failed",
                ))),
            }
        })
    }
}

#[test]
fn pending_mean_retires_permuted_reshape_aliases_without_releasing_input() {
    let provider = Provider {
        owner: TestAllocationOwner::new(),
        permutations: Mutex::new(Vec::new()),
        mode: Mode::Pending,
    };
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut future = Box::pin(provider_execution::reduce_mean_vecdim_nd_gpu(
        &provider,
        &input.handle,
        &[1],
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(provider.permutations.lock().unwrap()[0].retirements(), 0);
    drop(future);
    let permutations = provider.permutations.lock().unwrap();
    assert_eq!(permutations.len(), 1);
    assert_eq!(permutations[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn mean_only_unsupported_hooks_allow_fallback_and_execution_errors_retain_sources() {
    for mode in [Mode::Unsupported, Mode::Failure] {
        let provider = Provider {
            owner: TestAllocationOwner::new(),
            permutations: Mutex::new(Vec::new()),
            mode,
        };
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let result = futures::executor::block_on(provider_execution::execute(
            &provider,
            &input.handle,
            &MeanAxes::Dim(1),
            ReductionNaN::Include,
        ));
        match (mode, result) {
            (Mode::Unsupported, Ok(None)) => {}
            (Mode::Failure, Err(error)) => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert_eq!(
                    error
                        .source
                        .as_ref()
                        .unwrap()
                        .downcast_ref::<std::io::Error>()
                        .unwrap()
                        .kind(),
                    std::io::ErrorKind::BrokenPipe
                );
            }
            (_, actual) => panic!("unexpected mean provider outcome: {actual:?}"),
        }
        assert!(provider.permutations.lock().unwrap().is_empty());
        assert_eq!(input.probe.retirements(), 0);
    }
}
