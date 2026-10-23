use crate::catalog::entries::math::linalg::factor::inference::{
    double_scalar, factor_output, matrix_shape,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, ResidencyFact, StorageFact, ValueFact,
    ValueKindFact,
};
mod optional;

fn dimensions(value: &ValueFact) -> Option<(Option<usize>, Option<usize>)> {
    let dims = value.shape.known_dims()?;
    if dims.len() < 2 || dims.iter().skip(2).any(|&n| n != Some(1)) {
        return None;
    }
    Some((dims[0], dims[1]))
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=8).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM.GMRES.INVALID_INPUT",
            "gmres requires A and b and accepts at most six optional arguments",
            request.arguments.len().min(2),
        ));
    }
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let rhs = request
        .arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let callable = matches!(&matrix.kind, ValueKindFact::Callable(_));
    for (index, operand) in request.arguments.iter().enumerate() {
        if matches!(
            operand.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM.GMRES.UNSUPPORTED_INPUT",
                "gmres requires host-resident operands",
                index,
            ));
        }
    }
    if !callable
        && !matches!(
            matrix.storage,
            StorageFact::Scalar | StorageFact::Dense | StorageFact::Sparse | StorageFact::Unknown
        )
    {
        diagnostics.push(argument_error(
            "RM.GMRES.UNSUPPORTED_INPUT",
            "A must be a dense or CSC matrix, or a callable operator",
            0,
        ));
    }
    if !matches!(
        &matrix.kind,
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Single | NumericClass::Double)
    ) && !matches!(
        &matrix.kind,
        ValueKindFact::Callable(_) | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM.GMRES.UNSUPPORTED_INPUT",
            "A must have native single/double coefficients or be callable",
            0,
        ));
    }
    if !matches!(
        &rhs.kind,
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Single | NumericClass::Double)
    ) && !matches!(&rhs.kind, ValueKindFact::Unknown)
    {
        diagnostics.push(argument_error(
            "RM.GMRES.UNSUPPORTED_INPUT",
            "b must be a native single/double column vector",
            1,
        ));
    }
    if let (Some(a), Some(b)) = (matrix.numeric(), rhs.numeric()) {
        if a != b {
            diagnostics.push(argument_error(
                "RM.GMRES.UNSUPPORTED_INPUT",
                "A and b must have the same native precision and complex domain",
                1,
            ));
        }
    }
    let matrix_dimensions = if callable { None } else { dimensions(&matrix) };
    if !callable
        && matrix
            .shape
            .known_dims()
            .is_some_and(|dims| dims.iter().skip(2).any(|&n| n.is_some_and(|n| n != 1)))
    {
        diagnostics.push(argument_error(
            "RM.GMRES.INVALID_INPUT",
            "A must be a two-dimensional square matrix",
            0,
        ));
    }
    if rhs
        .shape
        .known_dims()
        .is_some_and(|dims| dims.iter().skip(2).any(|&n| n.is_some_and(|n| n != 1)))
        || dimensions(&rhs).is_some_and(|(_, cols)| cols.is_some_and(|cols| cols != 1))
    {
        diagnostics.push(argument_error(
            "RM.GMRES.INVALID_INPUT",
            "b must be a column vector",
            1,
        ));
    }
    if let Some((Some(rows), Some(cols))) = matrix_dimensions {
        if rows != cols {
            diagnostics.push(argument_error(
                "RM.GMRES.INVALID_INPUT",
                "A must be square",
                0,
            ));
        }
        if dimensions(&rhs).is_some_and(|(b_rows, _)| b_rows.is_some_and(|n| n != rows)) {
            diagnostics.push(argument_error(
                "RM.GMRES.INVALID_INPUT",
                "b must have size(A,1) rows",
                1,
            ));
        }
    }
    let rows = dimensions(&rhs)
        .and_then(|(rows, _)| rows)
        .or_else(|| matrix_dimensions.and_then(|(rows, _)| rows));
    optional::validate(request, &rhs, rows, &mut diagnostics);
    let mut solution = factor_output(&rhs, matrix_shape(rows, Some(1)), false, false, true);
    if let ValueKindFact::Numeric(numeric) = &rhs.kind {
        if matches!(numeric.class, NumericClass::Single | NumericClass::Double) {
            solution.kind = ValueKindFact::Numeric(*numeric);
        }
    }
    let residuals = factor_output(&rhs, matrix_shape(None, Some(1)), true, false, true);
    let mut iteration = double_scalar();
    iteration.shape = matrix_shape(Some(1), Some(2));
    iteration.storage = StorageFact::Dense;
    iteration.residency = ResidencyFact::Host;
    let result = finish_fixed_outputs(
        entry,
        request,
        vec![
            solution,
            double_scalar(),
            double_scalar(),
            iteration,
            residuals,
        ],
        diagnostics,
    );
    super::super::entry::iterative_effects(result, request, &[0, 5, 6], true)
}
