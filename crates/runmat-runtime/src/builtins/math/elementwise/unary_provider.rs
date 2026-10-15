use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, ProviderPrecision,
};
use runmat_value::{NumericDType, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(test)]
#[path = "unary_provider_owned_tests.rs"]
mod owned_tests;

pub(super) fn validate_real_unary_output(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    mut output: GpuTensorHandle,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let contract = gpu_helpers::UnaryGpuOutputContract {
        storage: GpuTensorStorage::Real,
        precision: runmat_accelerate_api::handle_precision(input),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !gpu_helpers::unary_gpu_output_matches(&output, input, provider, contract)
        || !gpu_helpers::expected_handle_numeric_element_type(&output).is_ok_and(|element| {
            gpu_helpers::numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
        })
    {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            internal(builtin, "provider returned malformed unary output"),
        ));
    }
    preserve_provenance(&mut output, input)?;
    gpu_helpers::resident_gpu_value(output)
}

pub(super) async fn gather_compute_restore<F>(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    builtin: &'static str,
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> BuiltinResult<Value>,
{
    let gathered = {
        let _metadata = gpu_helpers::ResidencyRestoreGuard::new(input);
        gpu_helpers::gather_value_async(&Value::GpuTensor(input.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?
    };
    let output = compute(gathered)?;
    restore_output(provider, input, output, builtin)
}

pub(super) fn restore_output(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: Value,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let output = crate::builtins::math::resident_real_unary::preserve_elementwise_scalar_shape(
        output,
        input.shape(),
        builtin,
    )?;
    let (mut handle, storage, precision) = match &output {
        Value::Tensor(tensor) => (
            gpu_helpers::upload_tensor_protecting(provider, tensor, &[input])
                .map_err(|error| retained_internal(builtin, error))?,
            GpuTensorStorage::Real,
            precision_for(tensor.numeric_dtype()),
        ),
        Value::ComplexTensor(tensor) => (
            gpu_helpers::upload_complex_tensor_protecting(provider, tensor, &[input])
                .map_err(|error| retained_internal(builtin, error))?,
            GpuTensorStorage::ComplexInterleaved,
            precision_for(tensor.numeric_dtype()),
        ),
        _ => {
            return Err(internal(
                builtin,
                "host fallback produced unsupported output",
            ))
        }
    };
    let contract = gpu_helpers::UnaryGpuOutputContract {
        storage,
        precision,
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !gpu_helpers::unary_gpu_output_matches(&handle, input, provider, contract) {
        return Err(gpu_helpers::reject_gpu_output(
            handle,
            internal(
                builtin,
                "provider upload returned malformed fallback output",
            ),
        ));
    }
    preserve_provenance(&mut handle, input)?;
    gpu_helpers::resident_gpu_value(handle)
}

fn preserve_provenance(output: &mut GpuTensorHandle, input: &GpuTensorHandle) -> BuiltinResult<()> {
    let provenance =
        runmat_accelerate_api::handle_provenance(input).unwrap_or(GpuHandleProvenance::Automatic);
    *output = output.with_provenance(provenance).map_err(|error| {
        build_runtime_error(format!("cannot preserve unary output intent: {error}"))
            .with_source(error)
            .build()
    })?;
    Ok(())
}

fn precision_for(dtype: NumericDType) -> Option<ProviderPrecision> {
    match dtype {
        NumericDType::F32 => Some(ProviderPrecision::F32),
        NumericDType::F64 => Some(ProviderPrecision::F64),
        _ => None,
    }
}

fn internal(builtin: &'static str, detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("{builtin}: {detail}"))
        .with_builtin(builtin)
        .build()
}

fn retained_internal(builtin: &'static str, error: RuntimeError) -> RuntimeError {
    let mut contextual = internal(builtin, error.message());
    contextual.source = Some(Box::new(error));
    contextual
}
