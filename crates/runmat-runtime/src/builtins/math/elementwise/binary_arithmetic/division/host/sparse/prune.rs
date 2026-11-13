//! Remove explicit zero quotient entries while preserving native CSC storage.

use super::DivisionContext;
use crate::BuiltinResult;
use runmat_value::SparseTensor;

pub(super) fn prune_result_zeros(
    context: DivisionContext,
    output: SparseTensor,
) -> BuiltinResult<SparseTensor> {
    let mut zeros = Vec::new();
    for col in 0..output.cols {
        for index in output.col_ptrs[col]..output.col_ptrs[col + 1] {
            let zero = output
                .numeric_value_at(index)
                .is_some_and(|value| value.is_zero())
                || output
                    .complex_value_at(index)
                    .is_some_and(|value| value.0 == 0.0 && value.1 == 0.0);
            if zero {
                let linear = col
                    .checked_mul(output.rows)
                    .and_then(|base| base.checked_add(output.row_indices[index]))
                    .ok_or_else(|| context.internal_error("sparse quotient index overflow"))?;
                zeros
                    .try_reserve(1)
                    .map_err(|error| context.internal_error(error))?;
                zeros.push(linear);
            }
        }
    }
    if zeros.is_empty() {
        return Ok(output);
    }
    output
        .rows
        .checked_mul(output.cols)
        .ok_or_else(|| context.internal_error("sparse quotient extent overflow"))?;
    let result = if output.is_complex() {
        let updates = zeros
            .into_iter()
            .map(|index| (index, (0.0, 0.0)))
            .collect::<Vec<_>>();
        output.with_updated_complex_linear_values(&updates)
    } else if let Some(storage) = output.integer_storage() {
        let zero = storage.zeros_like(1).value_at(0).expect("one typed zero");
        let updates = zeros
            .into_iter()
            .map(|index| (index, zero.clone()))
            .collect::<Vec<_>>();
        output.with_updated_integer_linear_values(&updates)
    } else if output.numeric_dtype() == Some(runmat_value::NumericDType::F32) {
        let updates = zeros
            .into_iter()
            .map(|index| (index, 0.0f32))
            .collect::<Vec<_>>();
        output.with_updated_f32_linear_values(&updates)
    } else {
        let updates = zeros
            .into_iter()
            .map(|index| (index, 0.0f64))
            .collect::<Vec<_>>();
        output.with_updated_linear_values(&updates)
    };
    result.map_err(|error| context.internal_error(error))
}
