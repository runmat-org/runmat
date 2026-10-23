use crate::catalog::entries::math::linalg::factor::inference::{
    dimensions, double_scalar, extent, factor_output, matrix_shape,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, ResidencyFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=7).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM.PCG.INVALID_INPUT",
            "pcg requires A and b, followed by at most five optional arguments",
            request.arguments.len().min(2),
        ));
    }
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let callable = matches!(&matrix.kind, ValueKindFact::Callable(_));
    let rhs = request
        .arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    for (index, operand) in [(0, &matrix), (1, &rhs)] {
        if matches!(
            operand.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM.PCG.UNSUPPORTED_INPUT",
                "pcg operands require host residency",
                index,
            ));
        }
    }
    if let (Some(coefficient), Some(vector)) = (matrix.numeric(), rhs.numeric()) {
        if coefficient != vector {
            diagnostics.push(argument_error(
                "RM.PCG.UNSUPPORTED_INPUT",
                "A and b must have matching floating precision and real/complex domain",
                1,
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
            "RM.PCG.UNSUPPORTED_INPUT",
            "pcg requires a dense or sparse host matrix, or a callable operator",
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
            "RM.PCG.UNSUPPORTED_INPUT",
            "A must have single or double numeric coefficients, or be callable",
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
            "RM.PCG.UNSUPPORTED_INPUT",
            "b must be a single or double numeric column vector",
            1,
        ));
    }
    if rhs.shape.rank().is_some_and(|rank| rank > 2)
        || dimensions(&rhs).is_some_and(|(_, columns)| columns.is_some_and(|value| value != 1))
    {
        diagnostics.push(argument_error(
            "RM.PCG.INVALID_INPUT",
            "b must be a column vector",
            1,
        ));
    }
    if !callable && matrix.shape.rank().is_some_and(|rank| rank > 2) {
        diagnostics.push(argument_error(
            "RM.PCG.INVALID_INPUT",
            "A must be a two-dimensional square matrix",
            0,
        ));
    }
    let matrix_dimensions = if callable { None } else { dimensions(&matrix) };
    if let Some((Some(rows), Some(columns))) = matrix_dimensions {
        if rows != columns {
            diagnostics.push(argument_error(
                "RM.PCG.INVALID_INPUT",
                "A must be square",
                0,
            ));
        }
        if dimensions(&rhs).is_some_and(|(rhs_rows, _)| rhs_rows.is_some_and(|value| value != rows))
        {
            diagnostics.push(argument_error(
                "RM.PCG.INVALID_INPUT",
                "b must have size(A,1) rows",
                1,
            ));
        }
    }
    let rows = extent(&rhs).or_else(|| matrix_dimensions.and_then(|(rows, _)| rows));
    let mut solution = factor_output(&rhs, matrix_shape(rows, Some(1)), false, false, true);
    // A successful solve preserves b's numeric class and complex domain.
    // Callable A carries no numeric output-class authority of its own.
    if let ValueKindFact::Numeric(numeric) = &rhs.kind {
        if matches!(numeric.class, NumericClass::Single | NumericClass::Double) {
            solution.kind = ValueKindFact::Numeric(*numeric);
        }
    }
    let residuals = factor_output(&rhs, matrix_shape(None, Some(1)), true, false, true);
    let result = finish_fixed_outputs(
        entry,
        request,
        vec![
            solution,
            double_scalar(),
            double_scalar(),
            double_scalar(),
            residuals,
        ],
        diagnostics,
    );
    super::super::entry::iterative_effects(result, request, &[0, 4, 5], false)
}
