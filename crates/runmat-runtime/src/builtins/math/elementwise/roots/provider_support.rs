use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, ProviderPrecision,
};
use runmat_value::{NumericDType, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

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
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
    };
    if !gpu_helpers::unary_gpu_output_matches(&output, input, provider, contract) {
        gpu_helpers::free_rejected_provider_output(&output, &[input], provider);
        return Err(internal(
            builtin,
            "provider returned malformed unary output",
        ));
    }
    preserve_provenance(&mut output, input);
    Ok(gpu_helpers::resident_gpu_value(output))
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
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(input.clone()))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, builtin))?;
    let output = compute(gathered)?;
    restore_output(provider, input, output, builtin)
}

fn restore_output(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: Value,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let (mut handle, storage, precision) = match &output {
        Value::Num(value) => {
            let tensor = runmat_value::Tensor::new(vec![*value], vec![1, 1])
                .map_err(|error| internal(builtin, error))?;
            (
                gpu_helpers::upload_tensor(provider, &tensor)
                    .map_err(|error| internal(builtin, error))?,
                GpuTensorStorage::Real,
                Some(ProviderPrecision::F64),
            )
        }
        Value::Tensor(tensor) => (
            gpu_helpers::upload_tensor(provider, tensor)
                .map_err(|error| internal(builtin, error))?,
            GpuTensorStorage::Real,
            precision_for(tensor.numeric_dtype()),
        ),
        Value::Complex(real, imag) => {
            let tensor = runmat_value::ComplexTensor::new(vec![(*real, *imag)], vec![1, 1])
                .map_err(|error| internal(builtin, error))?;
            (
                gpu_helpers::upload_complex_tensor(provider, &tensor)
                    .map_err(|error| internal(builtin, error))?,
                GpuTensorStorage::ComplexInterleaved,
                Some(ProviderPrecision::F64),
            )
        }
        Value::ComplexTensor(tensor) => (
            gpu_helpers::upload_complex_tensor(provider, tensor)
                .map_err(|error| internal(builtin, error))?,
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
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
    };
    if !gpu_helpers::unary_gpu_output_matches(&handle, input, provider, contract) {
        gpu_helpers::free_rejected_provider_output(&handle, &[input], provider);
        return Err(internal(
            builtin,
            "provider upload returned malformed fallback output",
        ));
    }
    preserve_provenance(&mut handle, input);
    Ok(gpu_helpers::resident_gpu_value(handle))
}

fn preserve_provenance(output: &mut GpuTensorHandle, input: &GpuTensorHandle) {
    let provenance =
        runmat_accelerate_api::handle_provenance(input).unwrap_or(GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(output, provenance);
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
