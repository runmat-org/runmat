use super::{
    parameters, physical, plan, AxisPlan, GpuCapabilityReason, HostNetworkPlan, ParameterRole,
};
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use crate::builtins::deep_learning::{
    layers::{feature_input_layer_builtin, layer_normalization_layer_builtin, relu_layer_builtin},
    model::dlnetwork_builtin,
    object,
};
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, DeepLearningFeedForwardCapability,
    DeepLearningFeedForwardStage, DeepLearningForwardCapability, DeepLearningForwardResources,
    DeepLearningForwardRoute, DeepLearningLayerNormCapability, DeepLearningLayerNormRequest,
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

fn normalization_then_relu() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let normalization = block_on(layer_normalization_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, normalization, relu], 3, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn host_origin_uses_the_same_pure_physical_stage_lowering() {
    use runmat_accelerate_api::{DeepLearningForwardSource, NumericElementType};

    let network_owner = normalization_then_relu();
    let Value::Object(network) = network_owner.value() else {
        panic!("network object expected")
    };
    let input = Value::Tensor(
        runmat_value::Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
    );
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let lowered = plan::lower(network, &host, host.output_axes(), "forward")
        .unwrap()
        .expect("normalization plus activation has a mixed stage route")
        .expect("admitted host geometry must lower");
    assert!(lowered
        .stages
        .iter()
        .any(|stage| matches!(stage, plan::Stage::Normalize { .. })));
    assert!(lowered
        .stages
        .iter()
        .any(|stage| matches!(stage, plan::Stage::Relu)));
    let route = physical::from_plan(&host, &lowered, "forward")
        .unwrap()
        .expect("admitted parameters have a physical route");
    assert_eq!(route.analyze().unwrap().output_bytes, 16);

    let DeepLearningForwardSource::Host {
        element_type: NumericElementType::F32,
        shape,
    } = route.input
    else {
        panic!("admitted host input must stay a physical host source")
    };
    assert_eq!(shape, vec![2, 2]);
}

fn replace_normalization_parameter(
    network: &mut runmat_value::ObjectInstance,
    role: ParameterRole,
    value: Value,
) {
    let Value::Cell(layers) = network.fixture_dl_property_mut("Layers").unwrap() else {
        unreachable!()
    };
    let Value::Object(layer) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
    else {
        unreachable!()
    };
    layer
        .replace_dl_property(role.name(), value)
        .expect("fixture property");
}

#[test]
fn mixed_normalization_admits_resident_parameters_before_effects() {
    struct LaterStageProbe {
        device: u32,
        effects: AtomicUsize,
        reject_relu: AtomicBool,
        reject_route: AtomicBool,
        whole_route_available: AtomicBool,
        whole_route_queries: AtomicUsize,
        affine_operands: AtomicUsize,
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
            scale: Option<&GpuTensorHandle>,
            offset: Option<&GpuTensorHandle>,
            _: &DeepLearningLayerNormRequest,
        ) -> anyhow::Result<DeepLearningLayerNormCapability> {
            for parameter in [scale, offset].into_iter().flatten() {
                parameter.allocation_lease().validate_live()?;
                assert_eq!(parameter.shape(), &[2, 1]);
                assert_eq!(parameter.device_id(), self.device);
            }
            self.affine_operands.store(
                usize::from(scale.is_some()) + usize::from(offset.is_some()),
                Ordering::SeqCst,
            );
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
        fn deep_learning_forward_route_capability(
            &self,
            route: &DeepLearningForwardRoute<'_>,
        ) -> anyhow::Result<Option<DeepLearningForwardCapability>> {
            if !self.whole_route_available.load(Ordering::SeqCst) {
                return Ok(None);
            }
            let geometry = route
                .analyze()
                .map_err(|error| anyhow::anyhow!("{error:?}"))?;
            self.whole_route_queries.fetch_add(1, Ordering::SeqCst);
            Ok(Some(if self.reject_route.load(Ordering::SeqCst) {
                DeepLearningForwardCapability::UnsupportedOperation
            } else {
                DeepLearningForwardCapability::Ready(DeepLearningForwardResources {
                    output_bytes: geometry.output_bytes,
                    largest_stage_value_bytes: geometry.largest_stage_value_bytes,
                    known_normalization_scratch_bytes: 0,
                    upload_bytes: 0,
                    resident_bytes: 0,
                    new_buffer_bytes_upper_bound:
                        runmat_accelerate_api::DeepLearningForwardBufferBound::Unavailable,
                })
            }))
        }
    }

    let network_owner = normalization_then_relu();
    let Value::Object(network) = network_owner.value() else {
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
        reject_route: AtomicBool::new(false),
        whole_route_available: AtomicBool::new(true),
        whole_route_queries: AtomicUsize::new(0),
        affine_operands: AtomicUsize::new(0),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = object(
        "dlarray",
        vec![
            ("Data", Value::GpuTensor(allocated.handle.clone())),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    )
    .expect("fixture object");
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let result = plan::admit(network, &host, host.output_axes(), "forward").unwrap();
    assert!(matches!(
        result,
        Some(Err(GpuCapabilityReason::ProviderOperation))
    ));
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(allocated.probe.retirements(), 0);

    provider.reject_relu.store(false, Ordering::SeqCst);
    let ready = plan::admit(network, &host, host.output_axes(), "forward").unwrap();
    let Some(Ok(ready)) = ready else {
        panic!("the same fully admitted mixed route should be ready")
    };
    assert_eq!(ready.stages.len(), 2);
    assert_eq!(ready.output_axes, *host.output_axes());
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(provider.whole_route_queries.load(Ordering::SeqCst), 1);

    provider.reject_route.store(true, Ordering::SeqCst);
    assert!(matches!(
        plan::admit(network, &host, host.output_axes(), "forward").unwrap(),
        Some(Err(GpuCapabilityReason::ProviderOperation))
    ));
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    provider.reject_route.store(false, Ordering::SeqCst);
    provider
        .whole_route_available
        .store(false, Ordering::SeqCst);
    assert!(matches!(
        plan::admit(network, &host, host.output_axes(), "forward").unwrap(),
        Some(Ok(_))
    ));
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    provider.whole_route_available.store(true, Ordering::SeqCst);

    use runmat_accelerate_api::{GpuHandleProvenance, NumericElementType};
    let scale = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let offset = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut resident_network = network.clone();
    for (role, allocation) in [
        (ParameterRole::Scale, &scale),
        (ParameterRole::Offset, &offset),
    ] {
        replace_normalization_parameter(
            &mut resident_network,
            role,
            Value::GpuTensor(
                allocation
                    .handle
                    .clone()
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .unwrap(),
            ),
        );
    }
    let host = HostNetworkPlan::admit(&resident_network, &input, "forward").unwrap();
    let Some(Ok(ready)) =
        plan::admit(&resident_network, &host, host.output_axes(), "forward").unwrap()
    else {
        panic!("resident normalization parameters must reach planned admission")
    };
    assert_eq!(provider.affine_operands.load(Ordering::SeqCst), 2);
    assert_eq!(ready.consumed_resident_parameters.len(), 2);
    assert!(ready
        .consumed_resident_parameters
        .iter()
        .all(runmat_accelerate_api::handle_is_explicit));
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);

    let double_scale = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for explicit in [false, true] {
        let mut mismatch = resident_network.clone();
        let handle = if explicit {
            double_scale
                .handle
                .clone()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap()
        } else {
            double_scale.handle.clone()
        };
        replace_normalization_parameter(
            &mut mismatch,
            ParameterRole::Scale,
            Value::GpuTensor(handle),
        );
        let host = HostNetworkPlan::admit(&mismatch, &input, "forward").unwrap();
        provider.affine_operands.store(0, Ordering::SeqCst);
        let result = plan::admit(&mismatch, &host, host.output_axes(), "forward");
        if explicit {
            assert!(result.is_err(), "explicit precision cannot silently gather");
        } else {
            assert!(matches!(
                result.unwrap(),
                Some(Err(GpuCapabilityReason::ParameterPrecision(
                    ParameterRole::Scale
                )))
            ));
        }
        assert_eq!(provider.affine_operands.load(Ordering::SeqCst), 0);
        assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    }
    let malformed = owner
        .allocate_zeroed(vec![3, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    replace_normalization_parameter(
        &mut resident_network,
        ParameterRole::Scale,
        Value::GpuTensor(malformed.handle.clone()),
    );
    assert!(HostNetworkPlan::admit(&resident_network, &input, "forward").is_err());
    assert_eq!(provider.affine_operands.load(Ordering::SeqCst), 0);
    assert_eq!(provider.effects.load(Ordering::SeqCst), 0);
    assert_eq!(malformed.probe.retirements(), 0);
    for allocation in [&allocated, &scale, &offset, &double_scale] {
        assert_eq!(allocation.probe.retirements(), 0);
    }
}
