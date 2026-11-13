//! Provider layouts for consumed fully connected parameters.
use super::super::precision::project_owned;
use super::*;
use crate::builtins::common::gpu_helpers::{
    reject_gpu_output, unary_gpu_output_matches_shape, upload_tensor_protecting_typed,
    GpuOutputAliasPolicy, UnaryGpuOutputContract,
};

#[cfg(test)]
mod tests;

pub(super) async fn prepare_weights(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    layer: &ObjectInstance,
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    if let Some(handle) = matching_resident_parameter(layer, ParameterRole::Weights, input) {
        let matrix = if handle.shape().len() == 1 {
            let shape = [1, handle.shape()[0]];
            let view = provider.reshape(handle, &shape).map_err(|error| {
                provider_failure(function, "resident weight reshape", error.into())
            })?;
            validate_parameter_result(
                view,
                handle,
                provider,
                &shape,
                GpuOutputAliasPolicy::AllowInput,
                function,
            )?
        } else {
            handle.clone()
        };
        let shape = [matrix.shape()[1], matrix.shape()[0]];
        let transposed = provider.transpose(&matrix).map_err(|error| {
            provider_failure(function, "resident weight transpose", error.into())
        })?;
        let alias = if matrix.shape().iter().any(|dimension| *dimension > 1) {
            GpuOutputAliasPolicy::RequireDistinct
        } else {
            GpuOutputAliasPolicy::AllowInput
        };
        return validate_parameter_result(transposed, &matrix, provider, &shape, alias, function);
    }
    let weights = prepare_host_parameter(layer, ParameterRole::Weights, function).await?;
    let transposed = transpose_2d(&weights, function)?;
    upload_parameter(provider, transposed, input, &[input], function)
}

pub(super) async fn prepare_bias(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    layer: &ObjectInstance,
    input: &GpuTensorHandle,
    weights: &GpuTensorHandle,
    width: usize,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let shape = [1, width];
    if let Some(handle) = matching_resident_parameter(layer, ParameterRole::Bias, input) {
        let row = provider
            .reshape(handle, &shape)
            .map_err(|error| provider_failure(function, "resident bias reshape", error.into()))?;
        return validate_parameter_result(
            row,
            handle,
            provider,
            &shape,
            GpuOutputAliasPolicy::AllowInput,
            function,
        );
    }
    let bias = prepare_host_parameter(layer, ParameterRole::Bias, function).await?;
    let row = Tensor::from_numeric_storage(
        bias.into_numeric_storage()
            .map_err(|error| deep_learning_error(function, error))?,
        shape.to_vec(),
    )
    .map_err(|error| deep_learning_error(function, error))?;
    upload_parameter(provider, row, input, &[input, weights], function)
}

fn matching_resident_parameter<'a>(
    layer: &'a ObjectInstance,
    role: ParameterRole,
    input: &GpuTensorHandle,
) -> Option<&'a GpuTensorHandle> {
    resident_parameter_handle(layer, role).filter(|handle| {
        runmat_accelerate_api::handle_precision(handle)
            == runmat_accelerate_api::handle_precision(input)
    })
}

/// Authenticate a canonical resident learnable before any parameter upload.
/// A host row remains `None`; explicit residency never becomes a gather path.
pub(super) fn admit_resident_parameter<'a>(
    layer: &'a ObjectInstance,
    role: ParameterRole,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    function: &'static str,
) -> BuiltinResult<Result<Option<&'a GpuTensorHandle>, GpuCapabilityReason>> {
    let Some(handle) = resident_parameter_handle(layer, role) else {
        return Ok(Ok(None));
    };
    let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| {
            admission::invalid_owner(function, admission::GpuOperand::Parameter(role))
        })?;
    if !runmat_accelerate_api::same_provider_instance(provider, owner) {
        return Err(admission::invalid_owner(
            function,
            admission::GpuOperand::Parameter(role),
        ));
    }
    if runmat_accelerate_api::handle_precision(handle)
        != runmat_accelerate_api::handle_precision(input)
    {
        if runmat_accelerate_api::handle_is_explicit(handle) {
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
    Ok(Ok(Some(handle)))
}

pub(super) async fn prepare_host_parameter(
    layer: &ObjectInstance,
    role: ParameterRole,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let name = role.name();
    let value = tensor_property_value(layer, name, function)?;
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        return Err(unsupported_error(
            function,
            format!("{function}: explicit resident {name} cannot be gathered to prepare a network parameter"),
        ));
    }
    let owned = crate::gather_if_needed_async(value).await?;
    crate::builtins::common::tensor::value_into_tensor_for(function, owned)
        .map_err(|error| deep_learning_error(function, format!("{function}: {error}")))
        .and_then(|tensor| normalize_numeric_tensor(tensor, function))
}

fn input_dtype(input: &GpuTensorHandle) -> NumericDType {
    match runmat_accelerate_api::handle_precision(input) {
        Some(ProviderPrecision::F32) => NumericDType::F32,
        Some(ProviderPrecision::F64) => NumericDType::F64,
        None => unreachable!("GPU forward admission requires floating precision"),
    }
}

pub(super) fn upload_parameter(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let converted = project_owned(tensor, input_dtype(input), function)?;
    upload_tensor_protecting_typed(provider, &converted, protected).map_err(|error| {
        provider_failure(
            function,
            "parameter upload",
            ProviderExecutionError::from_source(error),
        )
    })
}

fn validate_parameter_result(
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    alias: GpuOutputAliasPolicy,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    if unary_gpu_output_matches_shape(
        &output,
        input,
        provider,
        shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias,
        },
    ) {
        Ok(output)
    } else {
        Err(reject_gpu_output(
            output,
            crate::runtime_error::build_runtime_error(format!(
                "{function}: provider returned an invalid parameter layout descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}
