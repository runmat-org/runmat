use super::{LU_ERROR_INVALID_ARGUMENT, LU_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    binary64_output, dimensions, factor_output, literal_factor_option, matrix_input, matrix_shape,
};
use crate::catalog::inference::finish_fixed_outputs;
use crate::BuiltinCatalogEntry;
use crate::FactorOption;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = matrix_input(
        request,
        2,
        false,
        LU_ERROR_INVALID_ARGUMENT.code,
        LU_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let (rows, columns) = dimensions(&matrix).unwrap_or((None, None));
    let combined = factor_output(&matrix, matrix.shape.clone(), false, false, false);
    let lower = factor_output(&matrix, matrix_shape(rows, rows), false, false, false);
    let upper = factor_output(&matrix, matrix_shape(rows, columns), false, false, false);
    let pivot_mode = if request.arguments.len() == 1 {
        Some(FactorOption::Matrix)
    } else {
        literal_factor_option(request, 1)
    };
    let pivot = match pivot_mode {
        Some(FactorOption::Matrix) => {
            factor_output(&matrix, matrix_shape(rows, rows), true, false, false)
        }
        Some(FactorOption::Vector) => {
            binary64_output(&matrix, matrix_shape(rows, Some(1)), true, false)
        }
        _ => {
            let mut unknown =
                factor_output(&matrix, ShapeFact::Ranked { rank: 2 }, true, false, false);
            unknown.kind = ValueKindFact::Unknown;
            unknown
        }
    };
    let first = match request.outputs.requested.known_count() {
        Some(1) => combined,
        Some(_) => lower,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(entry, request, vec![first, upper, pivot], diagnostics)
}
