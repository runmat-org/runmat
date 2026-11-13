use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderPrecision};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::super::validation::valid_complex_binary_output;
use super::super::super::{
    broadcast_repetitions, resident_output_from_sources, ArithmeticProviderOperation,
};
use super::super::common::{provider_contract_error, provider_error};

pub(super) async fn try_complex_pair(
    operation: ArithmeticProviderOperation,
    precision: ProviderPrecision,
    provider: &dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let Some((shape, _, _)) = broadcast_repetitions(left.shape(), right.shape()) else {
        return Ok(None);
    };
    match operation.apply(provider, left, right).await {
        Ok(output)
            if valid_complex_binary_output(precision, &output, left, right, provider, &shape) =>
        {
            resident_output_from_sources(output, [left, right]).map(Some)
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            provider_contract_error(
                operation,
                "provider returned invalid complex arithmetic output",
            ),
        )),
        Err(error) if error.is_unsupported() => Ok(None),
        Err(error) => Err(provider_error(operation, error)),
    }
}
