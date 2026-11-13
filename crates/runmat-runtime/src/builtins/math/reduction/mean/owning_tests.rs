use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, GpuTensorStorage, HostTensorView, NumericElementType,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

struct Provider {
    owner: TestAllocationOwner,
    permutations: Mutex<Vec<RetirementProbe>>,
    reshapes: AtomicUsize,
    reductions: AtomicUsize,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Pending,
    Unsupported,
    Failure,
    WrongReshape(usize),
    WrongReduction,
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
    fn permute(
        &self,
        input: &GpuTensorHandle,
        order: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result = (|| -> anyhow::Result<GpuTensorHandle> {
            let shape = order.iter().map(|&axis| input.shape()[axis]).collect();
            let allocation = self.owner.allocate_zeroed(
                shape,
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.permutations.lock().unwrap().push(allocation.probe);
            Ok(allocation.handle)
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn reshape(&self, input: &GpuTensorHandle, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let stage = self.reshapes.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::WrongReshape(index) if index == stage) {
            let allocation = self.owner.allocate_zeroed(
                shape.to_vec(),
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )?;
            self.permutations.lock().unwrap().push(allocation.probe);
            return Ok(allocation.handle);
        }
        Ok(input.with_view(input.view().clone().with_shape(shape.to_vec()))?)
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            self.reductions.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, Mode::Unsupported) {
                return Ok(None);
            }
            let result: anyhow::Result<GpuTensorHandle> = async move {
                match self.mode {
                    Mode::Pending => futures::future::pending().await,
                    Mode::Unsupported => Err(
                        runmat_accelerate_api::unsupported_provider_operation("mean dimension"),
                    ),
                    Mode::Failure => Err(anyhow::Error::new(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "mean provider failed",
                    ))),
                    Mode::WrongReshape(_) | Mode::WrongReduction => {
                        let mut shape = input.shape().to_vec();
                        shape[axis] = 1;
                        let element = if matches!(self.mode, Mode::WrongReduction) {
                            NumericElementType::F32
                        } else {
                            NumericElementType::F64
                        };
                        let output =
                            self.owner
                                .allocate_zeroed(shape, element, GpuTensorStorage::Real)?;
                        self.permutations.lock().unwrap().push(output.probe);
                        Ok(output.handle)
                    }
                }
            }
            .await;
            result.map(Some).map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
}

#[test]
fn pending_mean_retires_permuted_reshape_aliases_without_releasing_input() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        permutations: Mutex::new(Vec::new()),
        reshapes: AtomicUsize::new(0),
        reductions: AtomicUsize::new(0),
        mode: Mode::Pending,
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut future = Box::pin(provider_execution::reduce_mean_vecdim_nd_gpu(
        provider,
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
            reshapes: AtomicUsize::new(0),
            reductions: AtomicUsize::new(0),
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
            &OutputTemplate::Default,
        ));
        match (mode, result) {
            (Mode::Unsupported, Ok(None)) => {}
            (Mode::Failure, Err(error)) => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                let mut source: &(dyn std::error::Error + 'static) = &error;
                loop {
                    if let Some(error) = source.downcast_ref::<std::io::Error>() {
                        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
                        break;
                    }
                    source = source.source().expect("typed provider cause retained");
                }
            }
            (_, actual) => panic!("unexpected mean provider outcome: {actual:?}"),
        }
        assert!(provider.permutations.lock().unwrap().is_empty());
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[test]
fn mean_validates_each_original_contract_intermediate_before_the_next_hook() {
    for mode in [
        Mode::WrongReshape(0),
        Mode::WrongReduction,
        Mode::WrongReshape(1),
        Mode::WrongReshape(2),
    ] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            permutations: Mutex::new(Vec::new()),
            reshapes: AtomicUsize::new(0),
            reductions: AtomicUsize::new(0),
            mode,
        }));
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let result = futures::executor::block_on(provider_execution::reduce_mean_vecdim_nd_gpu(
            provider,
            &input.handle,
            &[1],
        ));
        let Err(provider_execution::MeanProviderFailure::Execution(error)) = result else {
            panic!("malformed stage must be terminal");
        };
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            provider.reductions.load(Ordering::SeqCst),
            usize::from(!matches!(mode, Mode::WrongReshape(0)))
        );
        assert_eq!(
            provider.reshapes.load(Ordering::SeqCst),
            match mode {
                Mode::WrongReshape(index) => index + 1,
                _ => 1,
            }
        );
        assert!(provider
            .permutations
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 1));
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
