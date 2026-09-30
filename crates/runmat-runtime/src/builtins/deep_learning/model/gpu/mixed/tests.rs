use super::*;
use crate::builtins::deep_learning::{layers::*, model::*, object};
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, DeepLearningFeedForwardCapability,
    DeepLearningFeedForwardStage, DeepLearningLayerNormCapability, DeepLearningLayerNormRequest,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned,
    HostTensorView,
};
use runmat_value::{CellArray, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[test]
fn consumed_resident_learnables_preserve_explicit_intent_across_later_stages() {
    use crate::builtins::common::gpu_helpers::propagate_output_provenance;
    use crate::builtins::deep_learning::model::{
        parameters::{ParameterKey, ParameterRole, ParameterShape, ParameterView},
        plan::normalization::{Geometry, NormalizationOptions, NormalizationStage},
    };
    use runmat_accelerate_api::{handle_is_explicit, GpuHandleProvenance, NumericElementType};
    use runmat_builtins::{DeepLearningAxisLayout, DeepLearningNormalizationAxes};

    let owner = TestAllocationOwner::new();
    let allocate = || {
        owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap()
    };
    let input = allocate();
    let weights = allocate();
    let bias = allocate();
    let scale = allocate();
    let offset = allocate();
    let unused = allocate();
    let output = allocate();
    let weights_value = Value::GpuTensor(
        weights
            .handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap(),
    );
    let bias_value = Value::GpuTensor(
        bias.handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap(),
    );
    let scale_value = Value::GpuTensor(
        scale
            .handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap(),
    );
    let offset_value = Value::GpuTensor(
        offset
            .handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap(),
    );
    let unused_value = Value::GpuTensor(
        unused
            .handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap(),
    );
    let values = [
        weights_value,
        bias_value,
        scale_value,
        offset_value,
        unused_value,
    ];
    let roles = [
        (1, ParameterRole::Weights),
        (1, ParameterRole::Bias),
        (2, ParameterRole::Scale),
        (2, ParameterRole::Offset),
        (3, ParameterRole::Weights),
    ];
    let views = values
        .iter()
        .zip(roles)
        .map(|(value, (layer_index, role))| ParameterView {
            key: ParameterKey {
                layer_index,
                layer_name: format!("layer{layer_index}"),
                role,
            },
            value: Some(value),
            expected_shape: ParameterShape::Dimensions(vec![Some(2), Some(2)]),
        })
        .collect::<Vec<_>>();
    let axes = AxisPlan::new(DeepLearningAxisLayout::parse("CB", 2).unwrap(), &[2, 2]).unwrap();
    let geometry = Geometry::new(
        &axes,
        NormalizationOptions {
            epsilon: 1.0e-5,
            operation: DeepLearningNormalizationAxes::ChannelOnly,
        },
    )
    .unwrap();
    let stages = [
        plan::Stage::Dense { layer_index: 1 },
        plan::Stage::Normalize {
            stage: NormalizationStage {
                layer_index: 2,
                geometry,
            },
        },
        plan::Stage::Relu,
    ];
    for stage in [&stages[0], &stages[1]] {
        let selected = parameters::consumed_resident_handles(&views, std::slice::from_ref(stage));
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(handle_is_explicit));
        let mut individual_output = output.handle.clone();
        propagate_output_provenance(
            &mut individual_output,
            std::iter::once(&input.handle).chain(selected.iter()),
        )
        .unwrap();
        assert!(handle_is_explicit(&individual_output));
    }
    let consumed = parameters::consumed_resident_handles(&views, &stages);
    assert_eq!(consumed.len(), 4);
    assert!(consumed.iter().all(handle_is_explicit));
    assert!(!consumed
        .iter()
        .any(|handle| handle.shares_allocation_with(&unused.handle)));
    let mut result = output.handle.clone();
    propagate_output_provenance(
        &mut result,
        std::iter::once(&input.handle).chain(consumed.iter()),
    )
    .unwrap();
    assert!(handle_is_explicit(&result));
    for allocation in [&input, &weights, &bias, &scale, &offset, &unused, &output] {
        assert_eq!(allocation.probe.retirements(), 0);
    }
}

fn normalization_then_relu() -> Value {
    let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let normalization = block_on(layer_normalization_layer_builtin(vec![])).unwrap();
    let relu = block_on(relu_layer_builtin(vec![])).unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, normalization, relu], 3, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn later_single_relu_capability_rejects_mixed_route_before_effects() {
    struct LaterStageProbe {
        device: u32,
        effects: AtomicUsize,
        reject_relu: AtomicBool,
    }
    impl AccelProvider for LaterStageProbe {
        fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            self.effects.fetch_add(1, Ordering::SeqCst);
            panic!("capability admission must not upload")
        }
        fn download<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            self.effects.fetch_add(1, Ordering::SeqCst);
            panic!("capability admission must not download")
        }
        fn device_id(&self) -> u32 {
            self.device
        }
        fn device_info(&self) -> String {
            "later-stage-probe".into()
        }
        fn deep_learning_layer_norm_planned_capability(
            &self,
            _: &GpuTensorHandle,
            _: Option<&GpuTensorHandle>,
            _: Option<&GpuTensorHandle>,
            _: &DeepLearningLayerNormRequest,
        ) -> anyhow::Result<DeepLearningLayerNormCapability> {
            Ok(DeepLearningLayerNormCapability::Ready)
        }
        fn deep_learning_feed_forward_stage_capability(
            &self,
            _: &GpuTensorHandle,
            stage: DeepLearningFeedForwardStage,
        ) -> anyhow::Result<DeepLearningFeedForwardCapability> {
            Ok(
                if self.reject_relu.load(Ordering::SeqCst)
                    && matches!(stage, DeepLearningFeedForwardStage::Relu { .. })
                {
                    DeepLearningFeedForwardCapability::UnsupportedOperation
                } else {
                    DeepLearningFeedForwardCapability::Ready
                },
            )
        }
    }

    let Value::Object(network) = normalization_then_relu() else {
        panic!("network object expected")
    };
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let allocated = owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let provider = Box::leak(Box::new(LaterStageProbe {
        device: owner.device_id(),
        effects: AtomicUsize::new(0),
        reject_relu: AtomicBool::new(true),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = object(
        "dlarray",
        vec![
            ("Data", Value::GpuTensor(allocated.handle.clone())),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    );
    let host = HostNetworkPlan::admit(&network, &input, "forward").unwrap();
    let result = plan::admit(
        &network,
        &host,
        &allocated.handle,
        host.output_axes(),
        "forward",
    )
    .unwrap();
    assert!(matches!(
        result,
        Some(Err(GpuCapabilityReason::ProviderOperation))
    ));
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(allocated.probe.retirements(), 0);

    provider.reject_relu.store(false, Ordering::SeqCst);
    let ready = plan::admit(
        &network,
        &host,
        &allocated.handle,
        host.output_axes(),
        "forward",
    )
    .unwrap();
    let Some(Ok(ready)) = ready else {
        panic!("the same fully admitted mixed route should be ready")
    };
    assert_eq!(ready.stages.len(), 2);
    assert_eq!(ready.output_axes, *host.output_axes());
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
}
