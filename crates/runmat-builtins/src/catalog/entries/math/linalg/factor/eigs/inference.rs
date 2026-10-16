use super::{EIGS_ERROR_INVALID_ARGUMENT, EIGS_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    dimensions, double_scalar, extent, factor_output, matrix_input, matrix_shape,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = matrix_input(
        request,
        usize::MAX,
        true,
        EIGS_ERROR_INVALID_ARGUMENT.code,
        EIGS_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if let Some(other) = request.arguments.get(1) {
        if matches!(&other.kind, ValueKindFact::Numeric(_)) && !other.is_scalar() {
            if let (
                Some((Some(rows), Some(columns))),
                Some((Some(other_rows), Some(other_columns))),
            ) = (dimensions(&matrix), dimensions(other))
            {
                if rows != other_rows || columns != other_columns {
                    diagnostics.push(argument_error(
                        EIGS_ERROR_INVALID_INPUT.code,
                        "generalized eigenproblem matrices must have equal shape",
                        1,
                    ));
                }
            }
        }
    }
    let n = extent(&matrix);
    let vector = factor_output(&matrix, matrix_shape(None, Some(1)), false, true, true);
    let eigenvectors = factor_output(&matrix, matrix_shape(n, None), false, true, true);
    let diagonal = factor_output(&matrix, matrix_shape(None, None), false, true, true);
    let first = match request.outputs.requested.known_count() {
        Some(1) => vector,
        Some(_) => eigenvectors,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![first, diagonal, double_scalar()],
        diagnostics,
    )
}
