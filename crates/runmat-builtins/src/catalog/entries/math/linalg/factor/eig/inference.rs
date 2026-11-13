use super::{EIG_ERROR_INVALID_ARGUMENT, EIG_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    dimensions, extent, factor_output, literal_factor_option, matrix_input, matrix_shape,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use crate::FactorOption;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = matrix_input(
        request,
        3,
        true,
        EIG_ERROR_INVALID_ARGUMENT.code,
        EIG_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if let Some(other) = request.arguments.get(1) {
        if matches!(
            &other.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) {
            if let (
                Some((Some(rows), Some(columns))),
                Some((Some(other_rows), Some(other_columns))),
            ) = (dimensions(&matrix), dimensions(other))
            {
                if rows != other_rows || columns != other_columns {
                    diagnostics.push(argument_error(
                        EIG_ERROR_INVALID_INPUT.code,
                        "generalized eigenproblem matrices must have equal shape",
                        1,
                    ));
                }
            }
        }
    }
    let n = extent(&matrix);
    let eigenvalues = factor_output(&matrix, matrix_shape(n, Some(1)), false, true, false);
    let eigenvectors = factor_output(&matrix, matrix_shape(n, n), false, true, false);
    let first_option = if request.arguments.get(1).is_some_and(|other| {
        matches!(
            &other.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    }) {
        2
    } else {
        1
    };
    let mut selector = Some(FactorOption::Matrix);
    for index in first_option..request.arguments.len() {
        match literal_factor_option(request, index) {
            Some(FactorOption::Vector) => selector = Some(FactorOption::Vector),
            Some(FactorOption::Matrix) => selector = Some(FactorOption::Matrix),
            Some(FactorOption::Balance | FactorOption::NoBalance) => {}
            Some(_) | None => {
                selector = None;
                break;
            }
        }
    }
    let second_shape = match selector {
        Some(FactorOption::Vector) => matrix_shape(n, Some(1)),
        Some(FactorOption::Matrix) => matrix_shape(n, n),
        _ => ShapeFact::Ranked { rank: 2 },
    };
    let second = factor_output(&matrix, second_shape, false, true, false);
    let first = match request.outputs.requested.known_count() {
        Some(1) => eigenvalues,
        Some(_) => eigenvectors.clone(),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![first, second, eigenvectors],
        diagnostics,
    )
}
