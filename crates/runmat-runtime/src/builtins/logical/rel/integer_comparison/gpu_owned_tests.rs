use super::*;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelProvider, AccelProviderFuture, NumericElementType, ThreadProviderGuard,
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
enum Mode {
    Logical,
    Numeric,
    Unsupported,
    Failure,
    PendingSecondProjection,
}
struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    projections: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl Provider {
    fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            mode,
            projections: AtomicUsize::new(0),
            outputs: Mutex::new(Vec::new()),
        }))
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "comparison owner".into()
    }
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no upload in direct comparison")
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no download in direct comparison") })
    }
    fn elem_eq<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let output = match self.mode {
                Mode::Logical => self.owner.allocate_logical(vec![1, 0], vec![1, 2])?,
                Mode::Numeric => self.owner.allocate_zeroed(
                    vec![1, 2],
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )?,
                Mode::Unsupported => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "comparison unavailable",
                    )
                    .into())
                }
                Mode::Failure => return Err(anyhow::anyhow!("comparison execution failure").into()),
                Mode::PendingSecondProjection => unreachable!(),
            };
            self.outputs.lock().unwrap().push(output.probe);
            Ok(output.handle)
        })
    }
    fn unary_real<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.projections.fetch_add(1, Ordering::SeqCst) == 1 {
                return std::future::pending().await;
            }
            let output = self.owner.allocate_zeroed(
                vec![1, 2],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(output.probe);
            Ok(output.handle)
        })
    }
}

#[test]
fn comparison_adopts_only_logical_producer_and_preserves_borrowed_inputs() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    for mode in [Mode::Logical, Mode::Numeric] {
        let provider = Provider::new(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let right = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let result = block_on(try_gpu_equality_comparison(
            &left.handle,
            &right.handle,
            IntegerComparisonOp::Eq,
        ))
        .unwrap();
        if matches!(mode, Mode::Logical) {
            let Value::GpuTensor(result) = result.unwrap() else {
                panic!("resident comparison")
            };
            assert!(runmat_accelerate_api::handle_is_logical(&result));
            assert_eq!(result.physical().element_type(), NumericElementType::U8);
            assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
            drop(result);
        } else {
            assert!(result.is_err());
        }
        assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
        assert_eq!(left.probe.retirements(), 0);
        assert_eq!(right.probe.retirements(), 0);
    }
}

#[test]
fn comparison_fallback_requires_typed_unsupported() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    for mode in [Mode::Unsupported, Mode::Failure] {
        let provider = Provider::new(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let result = block_on(try_gpu_equality_comparison(
            &input.handle,
            &input.handle,
            IntegerComparisonOp::Eq,
        ));
        if matches!(mode, Mode::Unsupported) {
            assert!(result.is_none());
        } else {
            assert!(result
                .unwrap()
                .unwrap_err()
                .message()
                .contains("comparison execution failure"));
        }
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[test]
fn abandoning_second_projection_retires_first_owned_temporary() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new(Mode::PendingSecondProjection);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let left = provider
        .owner
        .allocate_zeroed(
            vec![1, 2],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    let right = provider
        .owner
        .allocate_zeroed(
            vec![1, 2],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    let mut future = Box::pin(try_gpu_ordering_comparison(
        &left.handle,
        &right.handle,
        IntegerComparisonOp::Lt,
    ));
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert_eq!(provider.projections.load(Ordering::SeqCst), 2);
    drop(future);
    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
    assert_eq!(left.probe.retirements(), 0);
    assert_eq!(right.probe.retirements(), 0);
}
