mod prepared;
mod range;

use runmat_value::{Tensor, Value};

use crate::{builtins::common::gpu_helpers, BuiltinResult};

use super::operands::{BoundOperand, RescaleInput};
use super::{broadcast, error, provider, BUILTIN_NAME};

pub(super) fn rescale(
    input: RescaleInput,
    lower: BoundOperand,
    upper: BoundOperand,
    input_min: BoundOperand,
    input_max: BoundOperand,
) -> BuiltinResult<Value> {
    let restore = input.resident_source.is_some()
        || lower.was_resident()
        || upper.was_resident()
        || input_min.was_resident()
        || input_max.was_resident();
    let source = gpu_helpers::select_resident_output_source(
        [
            input.resident_source.clone(),
            lower.resident_source.clone(),
            upper.resident_source.clone(),
            input_min.resident_source.clone(),
            input_max.resident_source.clone(),
        ]
        .into_iter()
        .flatten(),
        BUILTIN_NAME,
    )?;
    let shape = broadcast::output_shape(
        &input.tensor.shape,
        &[
            &lower.tensor.shape,
            &upper.tensor.shape,
            &input_min.tensor.shape,
            &input_max.tensor.shape,
        ],
    )?;
    let length = broadcast::element_count(&shape)?;
    if length == 0 {
        return provider::empty(shape, input.output_dtype, restore, source.as_ref());
    }

    let operands =
        prepared::PreparedOperands::new(&input, &lower, &upper, &input_min, &input_max, &shape);
    let mut output = Vec::with_capacity(length);
    for linear in 0..length {
        output.push(range::scale(operands.at(linear, &shape)?)?);
    }
    let output = output
        .into_iter()
        .map(|value| range::cast(value, input.output_dtype))
        .collect();
    let tensor =
        Tensor::new_with_dtype(output, shape, input.output_dtype).map_err(error::internal)?;
    provider::output(tensor, restore, source.as_ref())
}
