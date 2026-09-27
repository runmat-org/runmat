use super::*;
use crate::builtins::common::{spec::ReductionNaN, test_support};
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
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
enum Behavior {
    AliasInput,
    WrongShape,
    AliasIntermediate,
    PendingDownload,
    ForeignOwner,
    ComplexStorage,
}

struct ReductionProvider {
    owner: TestAllocationOwner,
    foreign: TestAllocationOwner,
    behavior: Behavior,
    created: Mutex<Vec<RetirementProbe>>,
    calls: AtomicUsize,
}

impl ReductionProvider {
    fn new(behavior: Behavior) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            foreign: TestAllocationOwner::new(),
            behavior,
            created: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        }
    }
    fn result(
        &self,
        input: &GpuTensorHandle,
        axis: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.behavior, Behavior::AliasInput)
            || (matches!(self.behavior, Behavior::AliasIntermediate) && axis == Some(1))
        {
            return Ok(input.clone());
        }
        let shape = if matches!(self.behavior, Behavior::WrongShape) {
            vec![1, 3]
        } else if let Some(axis) = axis {
            let mut shape = input.shape().to_vec();
            shape[axis] = 1;
            shape
        } else {
            vec![1, 1]
        };
        let owner = if matches!(self.behavior, Behavior::ForeignOwner) {
            &self.foreign
        } else {
            &self.owner
        };
        let storage = if matches!(self.behavior, Behavior::ComplexStorage) {
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        } else {
            runmat_accelerate_api::GpuTensorStorage::Real
        };
        let allocation = owner.allocate_zeroed(
            shape,
            runmat_accelerate_api::NumericElementType::F64,
            storage,
        )?;
        self.created.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn source(&self) -> TestAllocation {
        self.owner
            .allocate_zeroed(
                vec![2, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
    }
}

impl AccelProvider for ReductionProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned reduction fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("reduction does not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        if matches!(self.behavior, Behavior::PendingDownload) {
            Box::pin(std::future::pending())
        } else {
            Box::pin(async { panic!("invalid result must not reach download") })
        }
    }

    fn reduce_all<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, None) })
    }
    fn reduce_any<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, None) })
    }
    fn reduce_all_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, Some(axis)) })
    }
    fn reduce_any_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
        _: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(input, Some(axis)) })
    }
}

fn provider(behavior: Behavior) -> &'static ReductionProvider {
    Box::leak(Box::new(ReductionProvider::new(behavior)))
}

fn configs() -> [&'static LogicalReductionConfig; 2] {
    [
        &crate::builtins::math::reduction::all::CONFIG,
        &crate::builtins::math::reduction::any::CONFIG,
    ]
}

#[test]
fn foreign_owner_or_complex_storage_is_retired_before_next_dimension_hook() {
    let _state = test_support::accel_test_lock();
    for behavior in [Behavior::ForeignOwner, Behavior::ComplexStorage] {
        let provider = provider(behavior);
        let _provider = ThreadProviderGuard::set(Some(provider));
        for config in configs() {
            provider.created.lock().unwrap().clear();
            provider.calls.store(0, Ordering::SeqCst);
            let input = provider.source();
            let error = block_on(reduce(
                config,
                input.handle.clone(),
                ReductionSpec::Dimensions(vec![1, 2]),
                ReductionNaN::Include,
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), config.internal.identifier);
            assert_eq!(
                provider.calls.load(Ordering::SeqCst),
                1,
                "malformed first result must not reach the next hook"
            );
            assert!(input.handle.allocation_lease().validate_live().is_ok());
            assert_eq!(input.probe.retirements(), 0);
            let created = provider.created.lock().unwrap();
            assert_eq!(created.len(), 1);
            assert_eq!(
                created[0].retirements(),
                1,
                "rejected result releases its actual owner"
            );
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[test]
fn invalid_reduction_outputs_never_free_borrowed_inputs_and_free_fresh_results() {
    let _state = test_support::accel_test_lock();
    for behavior in [
        Behavior::AliasInput,
        Behavior::WrongShape,
        Behavior::AliasIntermediate,
    ] {
        let provider = provider(behavior);
        let _provider = ThreadProviderGuard::set(Some(provider));
        for config in configs() {
            provider.created.lock().unwrap().clear();
            let input = provider.source();
            let spec = if matches!(behavior, Behavior::AliasIntermediate) {
                ReductionSpec::Dimensions(vec![1, 2])
            } else {
                ReductionSpec::All
            };
            let error = block_on(reduce(
                config,
                input.handle.clone(),
                spec,
                ReductionNaN::Include,
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), config.internal.identifier);
            assert!(input.handle.allocation_lease().validate_live().is_ok());
            assert_eq!(input.probe.retirements(), 0);
            for output in provider.created.lock().unwrap().iter() {
                assert_eq!(output.retirements(), 1);
            }
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[test]
fn cancelling_reduction_download_releases_fresh_outputs_but_not_input() {
    let _state = test_support::accel_test_lock();
    let provider = provider(Behavior::PendingDownload);
    let _provider = ThreadProviderGuard::set(Some(provider));
    for config in configs() {
        provider.created.lock().unwrap().clear();
        let input = provider.source();
        let mut future = Box::pin(reduce(
            config,
            input.handle.clone(),
            ReductionSpec::All,
            ReductionNaN::Include,
        ));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(&noop_waker())),
            Poll::Pending
        ));
        drop(future);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(input.probe.retirements(), 0);
        let created = provider.created.lock().unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].retirements(), 1);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
