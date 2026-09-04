use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, ProviderPrecision,
};
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::operation::ExponentialOperation;
use super::contract;

pub(super) fn value(
    operation: ExponentialOperation,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    value: Value,
    provenance: GpuHandleProvenance,
) -> BuiltinResult<Value> {
    match value {
        Value::ComplexTensor(tensor) => complex(operation, provider, input, tensor, provenance),
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], input.shape.clone())
                .map_err(|error| super::super::errors::internal(operation, error))?;
            complex(operation, provider, input, tensor, provenance)
        }
        Value::Tensor(tensor) => real(
            operation,
            provider,
            input,
            tensor,
            contract::precision(input),
        ),
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], input.shape.clone())
                .map_err(|error| super::super::errors::internal(operation, error))?;
            real(
                operation,
                provider,
                input,
                tensor,
                contract::precision(input),
            )
        }
        value => Err(super::super::errors::internal(
            operation,
            format!("unexpected host fallback result {value:?}"),
        )),
    }
}

pub(super) fn real(
    operation: ExponentialOperation,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    tensor: Tensor,
    expected_precision: Option<ProviderPrecision>,
) -> BuiltinResult<Value> {
    let mut output = gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
        super::super::errors::internal(
            operation,
            format!("failed to restore fallback result to input provider: {error}"),
        )
    })?;
    if !contract::valid_real(&output, input, provider, expected_precision) {
        gpu_helpers::free_unprotected_exact_owner(&output, &[input]);
        return Err(super::super::errors::internal(
            operation,
            "provider upload returned malformed fallback output",
        ));
    }
    preserve_provenance(&mut output, input);
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn complex(
    operation: ExponentialOperation,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    tensor: ComplexTensor,
    provenance: GpuHandleProvenance,
) -> BuiltinResult<Value> {
    let mut output = gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|error| {
        super::super::errors::internal(
            operation,
            format!("failed to restore complex result to input provider: {error}"),
        )
    })?;
    let contract = gpu_helpers::UnaryGpuOutputContract {
        storage: GpuTensorStorage::ComplexInterleaved,
        precision: contract::precision(input),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
    };
    if !gpu_helpers::unary_gpu_output_matches(&output, input, provider, contract) {
        gpu_helpers::free_unprotected_exact_owner(&output, &[input]);
        return Err(super::super::errors::internal(
            operation,
            "provider upload returned malformed complex fallback output",
        ));
    }
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    Ok(gpu_helpers::complex_gpu_value(output))
}

fn preserve_provenance(output: &mut GpuTensorHandle, input: &GpuTensorHandle) {
    let provenance =
        runmat_accelerate_api::handle_provenance(input).unwrap_or(GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(output, provenance);
}
