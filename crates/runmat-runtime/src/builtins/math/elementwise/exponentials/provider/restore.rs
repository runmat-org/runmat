use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
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
) -> BuiltinResult<Value> {
    match value {
        Value::ComplexTensor(tensor) => complex(operation, provider, input, tensor),
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], input.shape().to_vec())
                .map_err(|error| super::super::errors::internal(operation, error))?;
            complex(operation, provider, input, tensor)
        }
        Value::Tensor(tensor) => real(
            operation,
            provider,
            input,
            tensor,
            contract::precision(input),
        ),
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], input.shape().to_vec())
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
    let output = gpu_helpers::upload_tensor_protecting(provider, &tensor, &[input])?;
    if !contract::valid_real(&output, input, provider, expected_precision) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            super::super::errors::internal(
                operation,
                "provider upload returned malformed fallback output",
            ),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

fn complex(
    operation: ExponentialOperation,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    tensor: ComplexTensor,
) -> BuiltinResult<Value> {
    let output = gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[input])?;
    let contract = gpu_helpers::UnaryGpuOutputContract {
        storage: GpuTensorStorage::ComplexInterleaved,
        precision: contract::precision(input),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !gpu_helpers::unary_gpu_output_matches(&output, input, provider, contract) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            super::super::errors::internal(
                operation,
                "provider upload returned malformed complex fallback output",
            ),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::complex_gpu_value(output)
}
