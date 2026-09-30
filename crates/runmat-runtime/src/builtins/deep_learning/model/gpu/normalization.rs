//! A resident stable normalization stage. Descriptor and whole-output-route
//! admission precede the first parameter upload or provider dispatch.

use super::admission::{invalid_owner, GpuOperand};
use super::*;
use crate::builtins::common::gpu_helpers::{
    propagate_output_provenance, reject_gpu_output, unary_gpu_output_matches_shape,
    GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use crate::builtins::deep_learning::model::plan::normalization::NormalizationStage;
use runmat_accelerate_api::DeepLearningLayerNormCapability;

fn parameter<'a>(layer: &'a ObjectInstance, role: ParameterRole) -> Option<&'a GpuTensorHandle> {
    resident_parameter_handle(layer, role)
}

async fn prepare_parameter(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    layer: &ObjectInstance,
    role: ParameterRole,
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    if let Some(handle) = parameter(layer, role) {
        return Ok(handle.clone());
    }
    let host = parameters::prepare_host_parameter(layer, role, function).await?;
    parameters::upload_parameter(provider, host, input, protected, function)
}

pub(in crate::builtins::deep_learning) async fn forward(
    network: &ObjectInstance,
    input: &GpuTensorHandle,
    stage: &NormalizationStage,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Result<GpuTensorHandle, GpuCapabilityReason>> {
    // A later composed stage/presentation plan can pre-admit device permutes.
    // This route deliberately declines them before any provider effect.
    if target_axes != &stage.geometry.axes {
        return Ok(Err(GpuCapabilityReason::AxisLayout));
    }
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| unsupported_error(function, "no provider owns the GPU dlarray input"))?;
    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        unreachable!("host plan admitted network Layers")
    };
    let Some(Value::Object(layer)) = layers.data.get(stage.layer_index) else {
        return Err(deep_learning_error(
            function,
            "admitted normalization stage no longer matches network Layers",
        ));
    };
    let scale = parameter(layer, ParameterRole::Scale);
    let offset = parameter(layer, ParameterRole::Offset);
    for (role, resident) in [
        (ParameterRole::Scale, scale),
        (ParameterRole::Offset, offset),
    ] {
        if let Some(resident) = resident {
            let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(resident)
                .ok_or_else(|| invalid_owner(function, GpuOperand::Parameter(role)))?;
            if !runmat_accelerate_api::same_provider_instance(provider, owner) {
                return Err(invalid_owner(function, GpuOperand::Parameter(role)));
            }
            if runmat_accelerate_api::handle_precision(resident)
                != runmat_accelerate_api::handle_precision(input)
            {
                if runmat_accelerate_api::handle_is_explicit(resident) {
                    return Err(unsupported_error(
                        function,
                        format!(
                            "{function}: explicit resident {} precision differs from input",
                            role.name()
                        ),
                    ));
                }
                return Ok(Err(GpuCapabilityReason::ParameterPrecision(role)));
            }
        }
    }
    let request = stage.geometry.resident_request();
    let capability = provider
        .deep_learning_layer_norm_capability(input, scale, offset, &request)
        .map_err(|error| {
            provider_failure(function, "resident normalization admission", error.into())
        })?;
    match capability {
        DeepLearningLayerNormCapability::Ready => {}
        DeepLearningLayerNormCapability::UnsupportedOperation => {
            return Ok(Err(GpuCapabilityReason::ProviderOperation))
        }
        DeepLearningLayerNormCapability::ResourceLimit => {
            return Ok(Err(GpuCapabilityReason::ResourceLimit))
        }
    }
    let scale = prepare_parameter(
        provider,
        layer,
        ParameterRole::Scale,
        input,
        &[input],
        function,
    )
    .await?;
    let offset = prepare_parameter(
        provider,
        layer,
        ParameterRole::Offset,
        input,
        &[input, &scale],
        function,
    )
    .await?;
    let output = provider
        .deep_learning_layer_norm(input, &scale, &offset, &request)
        .await
        .map_err(|error| {
            provider_failure(function, "resident layer normalization", error.into())
        })?;
    let mut output = validate_output(
        output,
        input,
        &scale,
        &offset,
        provider,
        stage.geometry.axes.shape(),
        function,
    )?;
    propagate_output_provenance(&mut output, [input, &scale, &offset])?;
    Ok(Ok(output))
}

fn validate_output(
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    scale: &GpuTensorHandle,
    offset: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let precision = runmat_accelerate_api::handle_precision(input);
    if !unary_gpu_output_matches_shape(
        &output,
        input,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) || output.shares_allocation_with(scale)
        || output.shares_allocation_with(offset)
    {
        return Err(reject_gpu_output(
            output,
            crate::build_runtime_error(format!(
                "{function}: provider returned an invalid normalization output descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::deep_learning::model::plan::normalization::Geometry;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{DeepLearningAxisLayout, DeepLearningNormalizationAxes};
    use runmat_value::{CellArray, Tensor};

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
            );
            let network = crate::builtins::deep_learning::object(
                "dlnetwork",
                vec![(
                    "Layers",
                    Value::Cell(CellArray::new(vec![Value::Num(0.0), layer], 2, 1).unwrap()),
                )],
            );
            let Value::Object(network) = network else {
                unreachable!()
            };
            provider.reset_telemetry();
            let stage = NormalizationStage {
                layer_index: 1,
                geometry,
            };
            let result = block_on(forward(&network, &input, &stage, &axes, "forward")).unwrap();
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
            );
            let network = crate::builtins::deep_learning::object(
                "dlnetwork",
                vec![(
                    "Layers",
                    Value::Cell(CellArray::new(vec![Value::Num(0.0), layer], 2, 1).unwrap()),
                )],
            );
            let Value::Object(network) = network else {
                unreachable!()
            };
            provider.reset_telemetry();
            let stage = NormalizationStage {
                layer_index: 1,
                geometry,
            };
            let error = block_on(forward(&network, &input, &stage, &axes, "forward")).unwrap_err();
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
}
