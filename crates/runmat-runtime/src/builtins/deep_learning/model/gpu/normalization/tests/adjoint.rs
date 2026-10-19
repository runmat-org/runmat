//! Reverse admission and publication preserve borrowed resident allocations.

use super::super::execution::NormalizationRecord;
use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelProvider, DeepLearningLayerNormCapability, DeepLearningLayerNormGradients,
    DeepLearningLayerNormRequest, GpuTensorHandle, GpuTensorStorage, NumericElementType,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum ReverseResult {
    Distinct,
    AliasedInput,
    CancelAfterReturn,
}

struct ReverseProbe {
    owner: TestAllocationOwner,
    cancellation: CancellationSource,
    result: ReverseResult,
    admissions: AtomicUsize,
    dispatches: AtomicUsize,
    results: Mutex<Vec<RetirementProbe>>,
}

impl AccelProvider for ReverseProbe {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "normalization-adjoint-ownership-probe".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("reverse admission must not upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("reverse admission must not gather or replay")
    }

    fn deep_learning_layer_norm_adjoint_capability(
        &self,
        anchor: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        statistics: Option<&GpuTensorHandle>,
        upstream: Option<&GpuTensorHandle>,
        _: &DeepLearningLayerNormRequest,
    ) -> anyhow::Result<DeepLearningLayerNormCapability> {
        self.admissions.fetch_add(1, Ordering::SeqCst);
        for handle in [Some(anchor), scale, offset, statistics, upstream]
            .into_iter()
            .flatten()
        {
            handle.allocation_lease().validate_live()?;
            if handle.device_id() != self.device_id() {
                anyhow::bail!("foreign reverse operand owner")
            }
        }
        Ok(DeepLearningLayerNormCapability::Ready)
    }

    fn deep_learning_layer_norm_adjoint<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        scale: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        request: &'a DeepLearningLayerNormRequest,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, DeepLearningLayerNormGradients> {
        Box::pin(async move {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            let dx = self.owner.allocate_zeroed(
                request.shape.clone(),
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            let dscale = self.owner.allocate_zeroed(
                request.scale_shape.clone(),
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            let doffset = self.owner.allocate_zeroed(
                request.offset_shape.clone(),
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            self.results
                .lock()
                .unwrap()
                .extend([dx.probe, dscale.probe, doffset.probe]);
            std::future::ready(()).await;
            if matches!(self.result, ReverseResult::CancelAfterReturn) {
                self.cancellation.cancel(CancellationReason::User);
            }
            Ok(DeepLearningLayerNormGradients {
                input: if matches!(self.result, ReverseResult::AliasedInput) {
                    scale.clone()
                } else {
                    dx.handle
                },
                scale: dscale.handle,
                offset: doffset.handle,
            })
        })
    }
}

struct BorrowedOperands {
    input: runmat_accelerate_api::test_support::TestAllocation,
    upstream: runmat_accelerate_api::test_support::TestAllocation,
    scale: runmat_accelerate_api::test_support::TestAllocation,
    offset: runmat_accelerate_api::test_support::TestAllocation,
    statistics: runmat_accelerate_api::test_support::TestAllocation,
}

impl BorrowedOperands {
    fn new(owner: &TestAllocationOwner) -> Self {
        let allocate = |shape| {
            owner
                .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap()
        };
        Self {
            input: allocate(vec![2, 2]),
            upstream: allocate(vec![2, 2]),
            scale: allocate(vec![2, 1]),
            offset: allocate(vec![2, 1]),
            statistics: allocate(vec![3, 2]),
        }
    }

    fn record(&self) -> NormalizationRecord {
        NormalizationRecord {
            layer_index: 1,
            scale: self.scale.handle.clone(),
            offset: self.offset.handle.clone(),
            statistics: self.statistics.handle.clone(),
            request: Geometry::new(
                &AxisPlan::new(DeepLearningAxisLayout::parse("CB", 2).unwrap(), &[2, 2]).unwrap(),
                crate::builtins::deep_learning::model::plan::normalization::NormalizationOptions {
                    epsilon: 1e-5,
                    operation: DeepLearningNormalizationAxes::ChannelOnly,
                },
            )
            .unwrap()
            .resident_request(),
        }
    }

    fn assert_live(&self) {
        for probe in [
            &self.input.probe,
            &self.upstream.probe,
            &self.scale.probe,
            &self.offset.probe,
            &self.statistics.probe,
        ] {
            assert_eq!(probe.retirements(), 0);
        }
    }
}

fn fixture(result: ReverseResult) -> (&'static ReverseProbe, BorrowedOperands) {
    let provider: &'static ReverseProbe = Box::leak(Box::new(ReverseProbe {
        owner: TestAllocationOwner::new(),
        cancellation: CancellationSource::new(),
        result,
        admissions: AtomicUsize::new(0),
        dispatches: AtomicUsize::new(0),
        results: Mutex::new(Vec::new()),
    }));
    let borrowed = BorrowedOperands::new(&provider.owner);
    (provider, borrowed)
}

#[test]
fn malformed_reverse_geometry_is_rejected_before_capability_or_dispatch() {
    let (provider, borrowed) = fixture(ReverseResult::Distinct);
    let malformed = provider
        .owner
        .allocate_zeroed(vec![1, 4], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let check = || Ok(());
    let error = block_on(super::super::adjoint::adjoint(
        provider,
        &borrowed.input.handle,
        &malformed.handle,
        &borrowed.record(),
        &check,
        "dlgradient",
    ))
    .unwrap_err();
    assert!(error.message().contains("geometry changed"));
    assert_eq!(provider.admissions.load(Ordering::SeqCst), 0);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    assert_eq!(malformed.probe.retirements(), 0);
    borrowed.assert_live();
}

#[test]
fn foreign_reverse_upstream_is_rejected_before_dispatch() {
    let (provider, borrowed) = fixture(ReverseResult::Distinct);
    let foreign = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let check = || Ok(());
    let error = block_on(super::super::adjoint::adjoint(
        provider,
        &borrowed.input.handle,
        &foreign.handle,
        &borrowed.record(),
        &check,
        "dlgradient",
    ))
    .unwrap_err();
    assert!(error.message().contains("foreign reverse operand owner"));
    assert_eq!(provider.admissions.load(Ordering::SeqCst), 1);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    assert_eq!(foreign.probe.retirements(), 0);
    borrowed.assert_live();
}

#[test]
fn aliased_reverse_result_retires_only_the_returned_reference() {
    let (provider, borrowed) = fixture(ReverseResult::AliasedInput);
    let check = || Ok(());
    let error = block_on(super::super::adjoint::adjoint(
        provider,
        &borrowed.input.handle,
        &borrowed.upstream.handle,
        &borrowed.record(),
        &check,
        "dlgradient",
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:gpu:ProviderPayloadMismatch")
    );
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
    borrowed.assert_live();
    let results = provider.results.lock().unwrap();
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|probe| probe.retirements() == 1));
}

#[test]
fn cancellation_after_reverse_dispatch_retires_only_new_results() {
    let (provider, borrowed) = fixture(ReverseResult::CancelAfterReturn);
    let token = provider.cancellation.token();
    let check = || {
        if token.is_cancelled() {
            Err(crate::runtime_error::semantic_error(
                "RunMat:Cancelled",
                "dlgradient: cancelled after normalization adjoint",
            ))
        } else {
            Ok(())
        }
    };
    let error = block_on(super::super::adjoint::adjoint(
        provider,
        &borrowed.input.handle,
        &borrowed.upstream.handle,
        &borrowed.record(),
        &check,
        "dlgradient",
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
    borrowed.assert_live();
    let results = provider.results.lock().unwrap();
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|probe| probe.retirements() == 1));
}
