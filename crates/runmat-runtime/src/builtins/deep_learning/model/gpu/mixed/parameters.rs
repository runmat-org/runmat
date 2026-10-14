//! Preflight exact resident parameter view chains before any network stage.

use super::super::*;
use crate::builtins::deep_learning::model::{
    admission::forward_data,
    parameters::{ParameterRole, ParameterView},
    plan::host::{Controls, HostNetworkPlan},
};
use runmat_accelerate_api::{
    ProviderViewCapability, ProviderViewChain, ProviderViewSource, ProviderViewStep,
};

pub(super) fn consumed_parameter_rows<'v, 'a>(
    views: &'v [ParameterView<'a>],
    stages: &[super::plan::Stage],
) -> Vec<&'v ParameterView<'a>> {
    let mut consumed = Vec::new();
    for stage in stages {
        let (layer_index, roles): (usize, &[ParameterRole]) = match stage {
            super::plan::Stage::Dense { layer_index } => {
                (*layer_index, &[ParameterRole::Weights, ParameterRole::Bias])
            }
            super::plan::Stage::Conv { layer_index, .. } => {
                (*layer_index, &[ParameterRole::Weights, ParameterRole::Bias])
            }
            super::plan::Stage::Normalize { stage } => (
                stage.layer_index,
                &[ParameterRole::Scale, ParameterRole::Offset],
            ),
            _ => continue,
        };
        for role in roles {
            let row = views
                .iter()
                .find(|row| row.key.layer_index == layer_index && row.key.role == *role)
                .expect("admitted learnable stage has a canonical parameter row");
            consumed.push(row);
        }
    }
    consumed
}

pub(super) fn consumed_resident_handles(
    views: &[ParameterView<'_>],
    stages: &[super::plan::Stage],
) -> Vec<GpuTensorHandle> {
    consumed_parameter_rows(views, stages)
        .into_iter()
        .filter_map(|row| match row.value.map(forward_data) {
            Some(Value::GpuTensor(handle)) => Some(handle.clone()),
            _ => None,
        })
        .collect()
}

pub(super) fn has_unsupported_resident_parameters(host: &HostNetworkPlan<'_>) -> bool {
    host.parameter_views().iter().any(|row| {
        if !row
            .value
            .is_some_and(|value| matches!(forward_data(value), Value::GpuTensor(_)))
        {
            return false;
        }
        !host.steps().iter().any(|step| {
            step.index == row.key.layer_index
                && matches!(
                    (&step.controls, row.key.role),
                    (
                        Controls::Dense,
                        ParameterRole::Weights | ParameterRole::Bias
                    ) | (
                        Controls::Convolution(_),
                        ParameterRole::Weights | ParameterRole::Bias
                    ) | (
                        Controls::Normalization(_),
                        ParameterRole::Scale | ParameterRole::Offset
                    )
                )
        })
    })
}

pub(super) fn admit_dense(
    host: &HostNetworkPlan<'_>,
    input: &GpuTensorHandle,
    layer_index: usize,
    input_channels: usize,
    output_channels: usize,
    function: &'static str,
) -> BuiltinResult<Result<(), GpuCapabilityReason>> {
    for role in [ParameterRole::Weights, ParameterRole::Bias] {
        let Some(row) = host
            .parameter_views()
            .iter()
            .find(|row| row.key.layer_index == layer_index && row.key.role == role)
        else {
            unreachable!("initialized dense layer has a canonical parameter row")
        };
        let Some(Value::GpuTensor(handle)) = row.value.map(forward_data) else {
            continue;
        };
        crate::builtins::common::gpu_helpers::exact_provider_for_binary_inputs(input, handle)
            .map_err(|_| {
                admission::invalid_owner(function, admission::GpuOperand::Parameter(role))
            })?;
        if runmat_accelerate_api::handle_precision(handle)
            != runmat_accelerate_api::handle_precision(input)
        {
            return Ok(Err(GpuCapabilityReason::ParameterPrecision(role)));
        }
        let steps = match role {
            ParameterRole::Weights => {
                let mut steps = Vec::with_capacity(2);
                if handle.shape().len() == 1 {
                    steps.push(ProviderViewStep::Reshape(vec![1, handle.shape()[0]]));
                }
                steps.push(ProviderViewStep::Transpose2D);
                steps
            }
            ParameterRole::Bias => vec![ProviderViewStep::Reshape(vec![1, output_channels])],
            _ => unreachable!("only consumed dense parameter roles are queried"),
        };
        let expected = match role {
            ParameterRole::Weights => [input_channels, output_channels],
            ParameterRole::Bias => [1, output_channels],
            _ => unreachable!("only consumed dense parameter roles are queried"),
        };
        let query = ProviderViewChain {
            source: ProviderViewSource::Current,
            steps,
        };
        let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
            .ok_or_else(|| admission::invalid_owner(function, admission::GpuOperand::Input))?;
        match provider
            .provider_view_chain_capability(handle, &query)
            .map_err(|error| {
                provider_failure(function, "resident parameter view admission", error.into())
            })? {
            ProviderViewCapability::Ready(outcome) if outcome.view.shape() == expected => {}
            ProviderViewCapability::Ready(_) => {
                return Err(deep_learning_error(
                    function,
                    "resident parameter view differs from admitted geometry",
                ));
            }
            ProviderViewCapability::UnsupportedOperation => {
                return Ok(Err(GpuCapabilityReason::ProviderOperation));
            }
            ProviderViewCapability::ResourceLimit => {
                return Ok(Err(GpuCapabilityReason::ResourceLimit));
            }
        }
    }
    Ok(Ok(()))
}
