use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use futures::executor::block_on;
use runmat_accelerate_api::{
    handle_provenance, GpuHandleProvenance, HostNumericDataView, HostNumericTensorView,
    HostTensorView,
};
use runmat_value::{CellArray, NumericDType, Tensor, Value};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{layers::*, model::*, object};

mod raw_inputs;

fn dense_network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn admitted_transpose_failure_is_operational_not_a_late_host_retry() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, AccelProvider, DeepLearningFeedForwardCapability,
        DeepLearningFeedForwardStage, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
        HostNumericTensorOwned,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailingTranspose {
        device: u32,
        effects: AtomicUsize,
    }
    impl AccelProvider for FailingTranspose {
        fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            panic!("admitted resident route must not upload")
        }

        fn download<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            panic!("operational transpose failure must not retry through a download")
        }

        fn device_info(&self) -> String {
            "post-admission transpose failure probe".into()
        }

        fn device_id(&self) -> u32 {
            self.device
        }
        fn deep_learning_feed_forward_stage_capability(
            &self,
            _anchor: &GpuTensorHandle,
            _stage: DeepLearningFeedForwardStage,
        ) -> anyhow::Result<DeepLearningFeedForwardCapability> {
            Ok(DeepLearningFeedForwardCapability::Ready)
        }
        fn transpose(&self, _input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
            self.effects.fetch_add(1, Ordering::SeqCst);
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "transpose capability changed after admission",
            ))
        }
    }

    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let allocated = owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: vec![2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let provider = Box::leak(Box::new(FailingTranspose {
        device: owner.device_id(),
        effects: AtomicUsize::new(0),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let network_owner = dense_network();
    let Value::Object(network) = network_owner.value() else {
        panic!("network object expected")
    };
    let input_axes = AxisPlan::new(
        runmat_builtins::DeepLearningAxisLayout::parse("CB", 2).unwrap(),
        &[2, 3],
    )
    .unwrap();
    let output_axes = AxisPlan::new(
        runmat_builtins::DeepLearningAxisLayout::parse("CB", 2).unwrap(),
        &[1, 3],
    )
    .unwrap();
    let error = block_on(super::evaluate_network_gpu(
        network,
        &allocated.handle,
        super::GpuForwardLayout::ChannelsBatch,
        super::NetworkOutputLayout::Preserve,
        &input_axes,
        &output_axes,
        &output_axes,
        None,
        "forward",
    ))
    .unwrap_err();
    assert!(error.to_string().contains("input layout transpose"));
    assert_eq!(provider.effects.load(Ordering::SeqCst), 1);
    assert_eq!(allocated.probe.retirements(), 0);
}

#[test]
fn explicit_nonsquare_cb_is_transposed_on_provider_and_restored_without_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let handle = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[2, 3],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle.clone())),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        )
        .expect("fixture object");
        provider.reset_telemetry();
        let network = dense_network();
        let Value::Object(result) = block_on(forward_builtin(
            network.value().clone(),
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("dlarray output expected")
        };
        let Some(Value::GpuTensor(output)) = result.properties().get("Data") else {
            panic!("resident output expected")
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(
            handle_provenance(output),
            Some(GpuHandleProvenance::Explicit)
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(
            result.properties().get("Format"),
            Some(&Value::String("CB".into()))
        );
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output.clone()))
                .unwrap();
        assert_eq!(gathered.materialize_f64(), vec![0.5, 2.5, 4.5]);
        let original =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle)).unwrap();
        assert_eq!(original.materialize_f64(), source);
    });
}

#[test]
fn single_cb_projects_double_parameters_once_and_keeps_native_resident_result() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let handle = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&source),
                shape: &[2, 3],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle)),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        )
        .expect("fixture object");
        provider.reset_telemetry();
        let network = dense_network();
        let Value::Object(result) = block_on(forward_builtin(
            network.value().clone(),
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("dlarray output expected")
        };
        let Some(Value::GpuTensor(output)) = result.properties().get("Data") else {
            panic!("resident output expected")
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(output),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output.clone()))
                .unwrap();
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        assert_eq!(gathered.materialize_f64(), vec![0.5, 2.5, 4.5]);
    });
}

#[test]
fn cb_softmax_reduces_channels_not_batches() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input_layer = block_on(feature_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let softmax = block_on(softmax_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let network = block_on(dlnetwork_builtin(vec![Value::Cell(
            CellArray::new(vec![input_layer, softmax], 2, 1).unwrap(),
        )]))
        .unwrap();
        let source = [1.0, 3.0, -2.0, 2.0, 0.0, 4.0];
        let host = object(
            "dlarray",
            vec![
                (
                    "Data",
                    Value::Tensor(Tensor::new(source.to_vec(), vec![2, 3]).unwrap()),
                ),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        )
        .expect("fixture object");
        let Value::Object(expected) = block_on(forward_builtin(
            network.value().clone(),
            host,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("host dlarray expected")
        };
        let Some(Value::Tensor(expected)) = expected.properties().get("Data") else {
            panic!("host tensor expected")
        };
        let handle = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[2, 3],
            })
            .unwrap();
        let resident = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle)),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        )
        .expect("fixture object");
        provider.reset_telemetry();
        let Value::Object(actual) = block_on(forward_builtin(
            network.value().clone(),
            resident,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("resident dlarray expected")
        };
        let Some(Value::GpuTensor(actual)) = actual.properties().get("Data") else {
            panic!("resident tensor expected")
        };
        assert_eq!(actual.shape(), &[2, 3]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(actual.clone()))
                .unwrap();
        for (left, right) in actual
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64())
        {
            assert!((left - right).abs() < 1.0e-12);
        }
    });
}

#[test]
fn malformed_transpose_result_rejects_only_returned_reference() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[2, 3],
            })
            .unwrap();
        provider.reset_telemetry();
        let error = super::validate_layout_transpose(input.clone(), &input, provider, "forward")
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let retained =
            crate::builtins::common::test_support::gather(Value::GpuTensor(input)).unwrap();
        assert_eq!(retained.materialize_f64(), source);
    });
}

#[test]
fn singleton_axis_transpose_accepts_a_valid_resident_view() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [2.0, 3.0, 5.0];
        let input = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[3, 1],
            })
            .unwrap();
        let view = provider.reshape(&input, &[1, 3]).unwrap();
        let output = super::validate_layout_transpose(view, &input, provider, "forward").unwrap();
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
        assert_eq!(gathered.shape(), &[1, 3]);
        assert_eq!(gathered.materialize_f64(), source);
    });
}

#[test]
fn nonsquare_transpose_accepts_only_a_correct_lazy_layout_on_shared_storage() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, AccelProvider, GpuTensorHandle, GpuTensorLayout,
        GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned, TransposeInfo,
    };
    // The in-process arithmetic provider admits contiguous views only. This
    // metadata-admission witness needs an owner that supports lazy views;
    // actual lazy payload execution is qualified separately on WGPU.
    struct ViewProbe(u32);
    impl AccelProvider for ViewProbe {
        fn device_id(&self) -> u32 {
            self.0
        }
        fn device_info(&self) -> String {
            "lazy-view-admission-probe".into()
        }
        fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            panic!("view admission must not upload")
        }
        fn download<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            panic!("view admission must not download")
        }
    }
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = TestAllocationOwner::new();
    let allocated = owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: vec![2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let provider = Box::leak(Box::new(ViewProbe(owner.device_id())));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = allocated.handle;
    let lazy = input
        .with_view(input.view().clone().with_shape(vec![3, 2]).with_layout(
            GpuTensorLayout::Transpose(TransposeInfo {
                base_rows: 2,
                base_cols: 3,
            }),
        ))
        .unwrap();
    let accepted = super::validate_layout_transpose(lazy, &input, provider, "forward").unwrap();
    assert!(accepted.shares_allocation_with(&input));

    let forged = input
        .with_view(input.view().clone().with_shape(vec![3, 2]))
        .unwrap();
    let error = super::validate_layout_transpose(forged, &input, provider, "forward").unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:gpu:ProviderPayloadMismatch")
    );
    input.allocation_lease().validate_live().unwrap();
    assert_eq!(allocated.probe.retirements(), 0);
}

#[test]
fn gpu_route_distinguishes_missing_kernel_from_invalid_feature_geometry() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: &[2, 3],
            })
            .unwrap();
        let axes = AxisPlan::new(
            runmat_builtins::DeepLearningAxisLayout::parse("BC", 2).unwrap(),
            input.shape(),
        )
        .unwrap();
        let sequence = block_on(sequence_input_layer_builtin(
            Value::Num(3.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let unsupported =
            super::admit_gpu_forward(&[sequence], &input, &axes, &axes, &axes, "forward").unwrap();
        assert!(matches!(
            unsupported,
            super::GpuForwardAdmission::UnsupportedCapability(
                super::GpuCapabilityReason::Layer(identity)
            ) if identity.is(runmat_builtins::DeepLearningLayerKind::SequenceInput.class())
        ));

        let feature = block_on(feature_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let relu = block_on(relu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        provider.reset_telemetry();
        let error =
            super::admit_gpu_forward(&[feature, relu], &input, &axes, &axes, &axes, "forward")
                .unwrap_err();
        assert!(error.message().contains("3 features"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn gpu_route_keeps_supported_fully_connected_execution_ready() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[2, 2],
            })
            .unwrap();
        let axes = AxisPlan::new(
            runmat_builtins::DeepLearningAxisLayout::parse("BC", 2).unwrap(),
            input.shape(),
        )
        .unwrap();
        let network_owner = dense_network();
        let Value::Object(network) = network_owner.value() else {
            panic!("network object expected")
        };
        let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
            panic!("network layers expected")
        };
        provider.reset_telemetry();
        assert!(matches!(
            super::admit_gpu_forward(layers.data(), &input, &axes, &axes, &axes, "forward")
                .unwrap(),
            super::GpuForwardAdmission::Ready(super::GpuForwardLayout::BatchChannels)
        ));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    });
}

#[test]
fn global_pool_requires_whole_route_capability_before_any_provider_effect() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = (1..=24).map(f64::from).collect::<Vec<_>>();
        let input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[2, 3, 4],
            })
            .unwrap();
        let axes = AxisPlan::new(
            runmat_builtins::DeepLearningAxisLayout::parse("CBT", 3).unwrap(),
            input.shape(),
        )
        .unwrap();
        let source = axes.after_global_average_pooling_1d().unwrap();
        let target = AxisPlan::new(
            runmat_builtins::DeepLearningAxisLayout::parse("BC", 2).unwrap(),
            &[3, 2],
        )
        .unwrap();
        let sequence = block_on(sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let pool = block_on(global_average_pooling_1d_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        provider.reset_telemetry();
        let result = super::admit_gpu_forward(
            &[sequence, pool],
            &input,
            &axes,
            &source,
            &target,
            "forward",
        )
        .unwrap();
        assert!(matches!(
            result,
            super::GpuForwardAdmission::UnsupportedCapability(
                super::GpuCapabilityReason::ProviderOperation
            )
        ));
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.upload_bytes, 0);
        assert_eq!(telemetry.download_bytes, 0);
    });
}

#[test]
fn gpu_capability_reason_keeps_identity_typed_until_diagnostic_rendering() {
    let identity = runmat_builtins::DeepLearningLayerKind::Convolution1D
        .class()
        .owned();
    let reason = super::GpuCapabilityReason::Layer(identity.clone());
    assert_eq!(reason, super::GpuCapabilityReason::Layer(identity));
    assert!(reason.to_string().contains("Convolution1DLayer"));
    assert_eq!(
        super::GpuCapabilityReason::ProviderOperation.to_string(),
        "provider does not implement a required resident operation"
    );
}

#[test]
fn incompatible_resident_parameter_owner_is_error_before_gpu_effects() {
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
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let axes = AxisPlan::new(
            runmat_builtins::DeepLearningAxisLayout::parse("BC", 2).unwrap(),
            input.shape(),
        )
        .unwrap();
        let feature = block_on(feature_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let dense = object_with_identity(
            FULLY_CONNECTED_LAYER_CLASS,
            vec![
                ("OutputSize", Value::Num(1.0)),
                ("Weights", Value::GpuTensor(foreign.handle.clone())),
                (
                    "Bias",
                    Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
                ),
            ],
        )
        .expect("fixture object");
        provider.reset_telemetry();
        let error =
            super::admit_gpu_forward(&[feature, dense], &input, &axes, &axes, &axes, "forward")
                .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(foreign.probe.retirements(), 0);

        // The public route's canonical parameter preflight is earlier than
        // GPU placement and owns the unsupported resident descriptor error.
        let network_owner = dense_network();
        let mut network = network_owner.value().clone();
        let Value::Object(network_object) = &mut network else {
            panic!("network object expected")
        };
        let Some(Value::Cell(layers)) = network_object.fixture_dl_property_mut("Layers") else {
            panic!("network layers expected")
        };
        let Value::Object(layer) =
            &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
        else {
            panic!("dense layer expected")
        };
        layer
            .replace_dl_property("Weights", Value::GpuTensor(foreign.handle.clone()))
            .expect("fixture property");
        let resident_input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(input)),
                ("Format", Value::String("BC".into())),
                ("Labels", Value::String("BC".into())),
            ],
        )
        .expect("fixture object");
        provider.reset_telemetry();
        let public_error = block_on(forward_builtin(
            network,
            resident_input,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(
            public_error.identifier(),
            Some("RunMat:deepLearning:InvalidInput")
        );
        assert!(public_error
            .message()
            .contains("Weights has an unsupported resident owner or class"));
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(foreign.probe.retirements(), 0);
    });
}

#[test]
fn malformed_unary_and_binary_results_retire_their_returned_lease() {
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
        let owner = TestAllocationOwner::new();
        let unary = owner
            .allocate_zeroed(
                vec![2, 2],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let unary_probe = unary.probe;
        let error = super::validate_unary_forward_result(unary.handle, &input, provider, "forward")
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(unary_probe.retirements(), 1);

        let binary = owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let binary_probe = binary.probe;
        let error = super::validate_binary_forward_result(
            binary.handle,
            &input,
            &input,
            provider,
            &[2, 2],
            runmat_accelerate_api::ProviderPrecision::F64,
            "forward",
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(binary_probe.retirements(), 1);
        let alias_error = super::validate_binary_forward_result(
            input.clone(),
            &input,
            &input,
            provider,
            &[2, 2],
            runmat_accelerate_api::ProviderPrecision::F64,
            "forward",
        )
        .unwrap_err();
        assert_eq!(
            alias_error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        let retained =
            crate::builtins::common::test_support::gather(Value::GpuTensor(input)).unwrap();
        assert_eq!(retained.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
    });
}
