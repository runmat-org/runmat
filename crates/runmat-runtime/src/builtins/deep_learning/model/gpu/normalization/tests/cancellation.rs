//! Recorded normalization cancellation retains failure provenance and leases.

use super::*;
use crate::builtins::deep_learning::model::gpu::stage_failure::StageExecutionFailure;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelProvider, DeepLearningLayerNormCapability, DeepLearningLayerNormRecorded,
    DeepLearningLayerNormRequest, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, NumericElementType,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct RecordedCancellationProvider {
    owner: TestAllocationOwner,
    cancellation: CancellationSource,
    fail_after_cancel: bool,
    uploads: AtomicUsize,
    dispatches: AtomicUsize,
    probes: Mutex<Vec<RetirementProbe>>,
}

impl AccelProvider for RecordedCancellationProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "recorded-normalization-cancellation".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("recorded native normalization must not use the floating upload")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let data = match host.data {
            HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
            HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
            _ => return Err(anyhow::anyhow!("fixture expects native floating parameters").into()),
        };
        let allocated = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.probes.lock().unwrap().push(allocated.probe);
        Ok(allocated.handle)
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("abandoned normalization must not gather or replay")
    }

    fn deep_learning_layer_norm_capability(
        &self,
        anchor: &GpuTensorHandle,
        _: Option<&GpuTensorHandle>,
        _: Option<&GpuTensorHandle>,
        _: &DeepLearningLayerNormRequest,
    ) -> anyhow::Result<DeepLearningLayerNormCapability> {
        anchor.allocation_lease().validate_live()?;
        Ok(DeepLearningLayerNormCapability::Ready)
    }

    fn deep_learning_layer_norm_recorded<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        request: &'a DeepLearningLayerNormRequest,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, DeepLearningLayerNormRecorded> {
        Box::pin(async move {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            let output = self.owner.allocate_zeroed(
                request.shape.clone(),
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            let statistics = self.owner.allocate_zeroed(
                vec![3, 2],
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            self.probes
                .lock()
                .unwrap()
                .extend([output.probe, statistics.probe]);
            std::future::ready(()).await;
            self.cancellation.cancel(CancellationReason::User);
            if self.fail_after_cancel {
                anyhow::bail!("recorded provider failed after cancellation")
            }
            Ok(DeepLearningLayerNormRecorded {
                output: output.handle,
                statistics: statistics.handle,
            })
        })
    }

    fn deep_learning_layer_norm<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        request: &'a DeepLearningLayerNormRequest,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            let output = self.owner.allocate_zeroed(
                request.shape.clone(),
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            self.probes.lock().unwrap().push(output.probe);
            std::future::ready(()).await;
            self.cancellation.cancel(CancellationReason::User);
            Ok(output.handle)
        })
    }
}

fn normalization_cancellation(fail_after_cancel: bool, recorded: bool) {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static RecordedCancellationProvider =
        Box::leak(Box::new(RecordedCancellationProvider {
            owner: TestAllocationOwner::new(),
            cancellation: CancellationSource::new(),
            fail_after_cancel,
            uploads: AtomicUsize::new(0),
            dispatches: AtomicUsize::new(0),
            probes: Mutex::new(Vec::new()),
        }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let axes = AxisPlan::new(
        DeepLearningAxisLayout::parse("CB", 2).unwrap(),
        input.handle.shape(),
    )
    .unwrap();
    let geometry = Geometry::new(
        &axes,
        crate::builtins::deep_learning::model::plan::normalization::NormalizationOptions {
            epsilon: 1e-5,
            operation: DeepLearningNormalizationAxes::ChannelOnly,
        },
    )
    .unwrap();
    let layer = crate::builtins::deep_learning::object_with_identity(
        runmat_builtins::DeepLearningLayerKind::LayerNormalization.class(),
        vec![
            (
                "Scale",
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
            ),
            (
                "Offset",
                Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap()),
            ),
        ],
    )
    .expect("fixture object");
    let Value::Object(network) = crate::builtins::deep_learning::object(
        "dlnetwork",
        vec![(
            "Layers",
            Value::Cell(CellArray::new(vec![Value::Num(0.0), layer], 2, 1).unwrap()),
        )],
    )
    .expect("fixture object") else {
        panic!("network fixture must be an object")
    };
    let stage = NormalizationStage {
        layer_index: 1,
        geometry,
    };
    let token = provider.cancellation.token();
    let check = || {
        if token.is_cancelled() {
            Err(crate::runtime_error::semantic_error(
                "RunMat:Cancelled",
                "forward: cancelled after recorded normalization dispatch",
            ))
        } else {
            Ok(())
        }
    };
    let failure = if recorded {
        match block_on(super::super::execution::forward_recorded(
            &network,
            &input.handle,
            &stage,
            &axes,
            &check,
            "forward",
        )) {
            Err(failure) => failure,
            Ok(_) => panic!("recorded normalization must not publish after cancellation"),
        }
    } else {
        match block_on(super::super::execution::forward(
            &network,
            &input.handle,
            &stage,
            &axes,
            &check,
            "forward",
        )) {
            Err(failure) => failure,
            Ok(_) => panic!("ordinary normalization must not publish after cancellation"),
        }
    };
    match failure {
        StageExecutionFailure::Abandoned(error) if !fail_after_cancel => {
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        }
        StageExecutionFailure::Operational(error) if fail_after_cancel => {
            assert!(error
                .message()
                .contains("provider failed after cancellation"));
        }
        _ => panic!("cancellation provenance or provider error precedence changed"),
    }
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
    assert_eq!(
        input.probe.retirements(),
        0,
        "borrowed input must remain live"
    );
    let probes = provider.probes.lock().unwrap();
    assert_eq!(probes.len(), if recorded { 4 } else { 3 });
    assert!(probes.iter().all(|probe| probe.retirements() == 1));
}

#[test]
fn recorded_normalization_cancellation_after_await_retires_owned_handles() {
    normalization_cancellation(false, true);
}

#[test]
fn recorded_normalization_provider_failure_precedes_concurrent_cancel() {
    normalization_cancellation(true, true);
}

#[test]
fn ordinary_normalization_cancellation_after_await_retires_owned_handles() {
    normalization_cancellation(false, false);
}
