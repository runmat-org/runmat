//! Provider layouts for consumed fully connected parameters.
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
    if let Some(handle) = matching_resident_parameter(layer, "Weights", input) {
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
    let weights = prepare_host_parameter(layer, "Weights", function).await?;
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
    if let Some(handle) = matching_resident_parameter(layer, "Bias", input) {
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
    let bias = prepare_host_parameter(layer, "Bias", function).await?;
    let row = floating_tensor_from_f64(
        tensor::tensor_values_f64(&bias),
        shape.to_vec(),
        input_dtype(input),
        function,
    )?;
    upload_parameter(provider, row, input, &[input, weights], function)
}

fn matching_resident_parameter<'a>(
    layer: &'a ObjectInstance,
    name: &str,
    input: &GpuTensorHandle,
) -> Option<&'a GpuTensorHandle> {
    resident_parameter_handle(layer, name).filter(|handle| {
        runmat_accelerate_api::handle_precision(handle)
            == runmat_accelerate_api::handle_precision(input)
    })
}

async fn prepare_host_parameter(
    layer: &ObjectInstance,
    name: &str,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    let value = tensor_property_value(layer, name, function)?;
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

fn upload_parameter(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let converted = floating_tensor_from_f64(
        tensor::tensor_values_f64(&tensor),
        tensor.shape,
        input_dtype(input),
        function,
    )?;
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
