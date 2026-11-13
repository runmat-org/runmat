use super::super::stage_failure::StageExecutionFailure;
use super::super::*;
use super::execution::forward;
use super::validation::validate_output;
use crate::builtins::deep_learning::model::plan::normalization::NormalizationStage;

use crate::builtins::deep_learning::model::plan::normalization::Geometry;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{DeepLearningAxisLayout, DeepLearningNormalizationAxes};
use runmat_value::{CellArray, Tensor};

#[path = "tests/cancellation.rs"]
mod cancellation;
#[path = "tests/adjoint.rs"]
mod adjoint;

#[test]
fn unsupported_provider_declines_before_upload_or_dispatch() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[2, 2],
            })
            .unwrap();
        let axes = AxisPlan::new(
            DeepLearningAxisLayout::parse("CB", 2).unwrap(),
            input.shape(),
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
        let network = crate::builtins::deep_learning::object(
            "dlnetwork",
            vec![(
                "Layers",
                Value::Cell(CellArray::new(vec![Value::Num(0.0), layer], 2, 1).unwrap()),
            )],
        )
        .expect("fixture object");
        let Value::Object(network) = network else {
            unreachable!()
        };
        provider.reset_telemetry();
        let stage = NormalizationStage {
            layer_index: 1,
            geometry,
        };
        let result = match block_on(forward(
            &network,
            &input,
            &stage,
            &axes,
            &|| Ok(()),
            "forward",
        )) {
            Ok(result) => result,
            Err(_) => panic!("unsupported provider must decline without an execution failure"),
        };
        assert_eq!(result.unwrap_err(), GpuCapabilityReason::ProviderOperation);
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.upload_bytes, 0);
        assert_eq!(telemetry.download_bytes, 0);
    });
}

#[test]
fn foreign_parameter_owner_is_rejected_before_provider_effects() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[2, 2],
            })
            .unwrap();
        let foreign = TestAllocationOwner::new()
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let axes = AxisPlan::new(
            DeepLearningAxisLayout::parse("CB", 2).unwrap(),
            input.shape(),
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
                ("Scale", Value::GpuTensor(foreign.handle.clone())),
                (
                    "Offset",
                    Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap()),
                ),
            ],
        )
        .expect("fixture object");
        let network = crate::builtins::deep_learning::object(
            "dlnetwork",
            vec![(
                "Layers",
                Value::Cell(CellArray::new(vec![Value::Num(0.0), layer], 2, 1).unwrap()),
            )],
        )
        .expect("fixture object");
        let Value::Object(network) = network else {
            unreachable!()
        };
        provider.reset_telemetry();
        let stage = NormalizationStage {
            layer_index: 1,
            geometry,
        };
        let error = block_on(forward(
            &network,
            &input,
            &stage,
            &axes,
            &|| Ok(()),
            "forward",
        ))
        .unwrap_err();
        let StageExecutionFailure::Operational(error) = error else {
            panic!("parameter owner mismatch must remain an operational failure");
        };
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.upload_bytes, 0);
        assert_eq!(telemetry.download_bytes, 0);
        assert_eq!(foreign.probe.retirements(), 0);
    });
}

#[test]
fn forged_output_aliasing_scale_retires_only_result_reference() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 3.0],
                shape: &[2, 1],
            })
            .unwrap();
        let scale = provider
            .upload(&HostTensorView {
                data: &[2.0, 4.0],
                shape: &[2, 1],
            })
            .unwrap();
        let offset = provider
            .upload(&HostTensorView {
                data: &[0.0, 0.0],
                shape: &[2, 1],
            })
            .unwrap();
        let rejected = validate_output(
            scale.clone(),
            &input,
            &scale,
            &offset,
            provider,
            &[2, 1],
            "forward",
        )
        .unwrap_err();
        assert_eq!(
            rejected.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(rejected.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let retained =
            crate::builtins::common::test_support::gather(Value::GpuTensor(scale)).unwrap();
        assert_eq!(retained.materialize_f64(), vec![2.0, 4.0]);
    });
}
