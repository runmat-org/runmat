//! Whole-route tape admission must reject an unsupported adjoint before any
//! callback, upload, dispatch, or borrowed-input retirement.

use super::{plan, GpuCapabilityReason, HostNetworkPlan};
use crate::builtins::deep_learning::{
    layers::{
        convolution_1d_layer_builtin, feature_input_layer_builtin,
        global_average_pooling_1d_layer_builtin, layer_normalization_layer_builtin,
        relu_layer_builtin, sequence_input_layer_builtin,
    },
    model::dlnetwork_builtin,
    object,
};
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider,
    DeepLearningConv1DCapability, DeepLearningConv1DOperands, DeepLearningConv1DRequest,
    DeepLearningFeedForwardCapability, DeepLearningFeedForwardStage,
    DeepLearningLayerNormCapability, DeepLearningLayerNormRequest, DeepLearningPool1DCapability,
    DeepLearningPool1DRoute, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
    HostNumericTensorOwned, HostTensorView,
};
use runmat_value::{CellArray, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

struct RejectAdjoint {
    device: u32,
    effects: AtomicUsize,
}

impl AccelProvider for RejectAdjoint {
    fn device_id(&self) -> u32 {
        self.device
    }

    fn device_info(&self) -> String {
        "resident-adjoint-preflight".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        panic!("unsupported adjoint must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        panic!("unsupported adjoint must not download")
    }

    fn deep_learning_feed_forward_stage_capability(
        &self,
        _: &GpuTensorHandle,
        _: DeepLearningFeedForwardStage,
    ) -> anyhow::Result<DeepLearningFeedForwardCapability> {
        Ok(DeepLearningFeedForwardCapability::Ready)
    }

    fn deep_learning_layer_norm_planned_capability(
        &self,
        anchor: &GpuTensorHandle,
        _: Option<&GpuTensorHandle>,
        _: Option<&GpuTensorHandle>,
        _: &DeepLearningLayerNormRequest,
    ) -> anyhow::Result<DeepLearningLayerNormCapability> {
        anchor.allocation_lease().validate_live()?;
        Ok(DeepLearningLayerNormCapability::Ready)
    }

    fn deep_learning_global_average_pool_1d_planned_capability(
        &self,
        anchor: &GpuTensorHandle,
        _: &[usize],
        _: &DeepLearningPool1DRoute,
    ) -> anyhow::Result<DeepLearningPool1DCapability> {
        anchor.allocation_lease().validate_live()?;
        Ok(DeepLearningPool1DCapability::Ready)
    }

    fn deep_learning_conv1d_capability(
        &self,
        operands: DeepLearningConv1DOperands<'_>,
        _: &DeepLearningConv1DRequest,
    ) -> anyhow::Result<DeepLearningConv1DCapability> {
        operands.anchor.allocation_lease().validate_live()?;
        Ok(DeepLearningConv1DCapability::UnsupportedOperation)
    }
}

fn network(layers: Vec<Value>) -> crate::execution::RootedValue {
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(layers, 3, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn resident_layers_reject_missing_reverse_capabilities_before_provider_effects() {
    let normalization = network(vec![
        block_on(feature_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        block_on(layer_normalization_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        block_on(relu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    ]);
    let pooling = network(vec![
        block_on(sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        block_on(global_average_pooling_1d_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        block_on(relu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    ]);
    let convolution = network(vec![
        block_on(sequence_input_layer_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        block_on(convolution_1d_layer_builtin(
            Value::Num(2.0),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
        block_on(relu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    ]);
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    static FIXTURE: OnceLock<(TestAllocationOwner, RejectAdjoint)> = OnceLock::new();
    let (owner, provider) = FIXTURE.get_or_init(|| {
        let owner = TestAllocationOwner::new();
        let provider = RejectAdjoint {
            device: owner.device_id(),
            effects: AtomicUsize::new(0),
        };
        (owner, provider)
    });
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for (network, data, shape, format) in [
        (normalization, vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], "CB"),
        (pooling, vec![1.0f32; 24], vec![2, 3, 4], "CBT"),
        (convolution, vec![1.0f32, 4.0], vec![1, 1, 2], "CBT"),
    ] {
        let allocated = owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(data),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(allocated.handle.clone())),
                ("Format", Value::String(format.into())),
                ("Labels", Value::String(format.into())),
            ],
        )
        .expect("fixture object");
        let Value::Object(network) = network.value() else {
            panic!("dlnetwork expected")
        };
        let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
        assert!(matches!(
            plan::admit_for_tape(network, &host, host.output_axes(), "forward").unwrap(),
            Some(Err(GpuCapabilityReason::ProviderOperation))
        ));
        assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
        assert_eq!(allocated.probe.retirements(), 0);
        allocated.handle.allocation_lease().validate_live().unwrap();
    }
}
