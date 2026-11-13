use super::{LU_ERROR_INVALID_ARGUMENT, LU_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    binary64_output, dimensions, factor_output, literal_factor_option, matrix_input, matrix_shape,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use crate::FactorOption;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    if request
        .arguments
        .first()
        .is_some_and(|a| a.storage == StorageFact::Sparse)
    {
        return sparse(request, entry);
    }
    let mut diagnostics = matrix_input(
        request,
        2,
        false,
        LU_ERROR_INVALID_ARGUMENT.code,
        LU_ERROR_INVALID_INPUT.code,
    );
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 3)
    {
        diagnostics.push(argument_error(
            LU_ERROR_INVALID_ARGUMENT.code,
            "four/five-output LU requires sparse coefficients",
            0,
        ));
    }
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

fn sparse(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = matrix_input(
        request,
        3,
        false,
        LU_ERROR_INVALID_ARGUMENT.code,
        LU_ERROR_INVALID_INPUT.code,
    );
    let matrix = &request.arguments[0];
    let (rows, columns) = dimensions(matrix).unwrap_or((None, None));
    let height = rows.zip(columns).map(|(m, n)| m.min(n));
    if !matches!(
        &matrix.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                ..
            })
    ) {
        diagnostics.push(argument_error(
            LU_ERROR_INVALID_INPUT.code,
            "sparse LU requires native single or double coefficients",
            0,
        ));
    }
    let mut vector = Some(false);
    let mut seen_format = false;
    for index in 1..request.arguments.len() {
        match literal_factor_option(request, index) {
            Some(FactorOption::Matrix) | Some(FactorOption::Vector) => {
                if seen_format {
                    diagnostics.push(argument_error(
                        LU_ERROR_INVALID_ARGUMENT.code,
                        "at most one permutation format is accepted",
                        index,
                    ));
                }
                seen_format = true;
                vector = Some(literal_factor_option(request, index) == Some(FactorOption::Vector));
            }
            Some(_) => diagnostics.push(argument_error(
                LU_ERROR_INVALID_ARGUMENT.code,
                "LU permutation form must be matrix or vector",
                index,
            )),
            None => {
                let value = &request.arguments[index];
                if matches!(&value.kind, ValueKindFact::Numeric(_)) {
                    if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
                        || matches!(
                            value.residency,
                            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
                        )
                    {
                        diagnostics.push(argument_error(
                            LU_ERROR_INVALID_ARGUMENT.code,
                            "pivot thresholds require host scalar or dense storage",
                            index,
                        ));
                    }
                    if index != 1 {
                        diagnostics.push(argument_error(
                            LU_ERROR_INVALID_ARGUMENT.code,
                            "pivot threshold precedes permutation format",
                            index,
                        ));
                    }
                    if !matches!(
                        &value.kind,
                        ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Single | NumericClass::Double,
                            domain: NumericDomain::Real
                        })
                    ) {
                        diagnostics.push(argument_error(
                            LU_ERROR_INVALID_ARGUMENT.code,
                            "sparse pivot threshold requires real floating storage",
                            index,
                        ));
                    }
                    if let Some(size) = value.shape.element_count() {
                        let many = request
                            .outputs
                            .requested
                            .known_count()
                            .is_none_or(|n| n >= 4);
                        if size != 1 && !(many && size == 2) {
                            diagnostics.push(argument_error(
                                LU_ERROR_INVALID_ARGUMENT.code,
                                "threshold requires one element, or two with four/five outputs",
                                index,
                            ));
                        }
                    }
                    if let Some(LiteralValue::Number(value)) =
                        request.literals.literal_args.get(index)
                    {
                        if !value.is_finite() || !(0.0..=1.0).contains(value) {
                            diagnostics.push(argument_error(
                                LU_ERROR_INVALID_ARGUMENT.code,
                                "pivot threshold must lie in [0,1]",
                                index,
                            ));
                        }
                    }
                } else {
                    vector = None;
                }
            }
        }
    }
    let native = |shape| {
        let mut value = factor_output(matrix, shape, false, false, true);
        value.storage = StorageFact::Sparse;
        if let ValueKindFact::Numeric(numeric) = &matrix.kind {
            value.kind = ValueKindFact::Numeric(*numeric);
        }
        value
    };
    let combined = native(matrix_shape(rows, columns));
    let lower = native(matrix_shape(rows, height));
    let upper = native(matrix_shape(height, columns));
    let permutation = |size| {
        let shape = match vector {
            Some(true) => matrix_shape(size, Some(1)),
            Some(false) => matrix_shape(size, size),
            None => ShapeFact::Ranked { rank: 2 },
        };
        let mut result = binary64_output(matrix, shape, true, true);
        result.storage = match vector {
            Some(true) => StorageFact::Dense,
            Some(false) => StorageFact::Sparse,
            None => StorageFact::Unknown,
        };
        result
    };
    let mut scaling = native(matrix_shape(rows, rows));
    if let ValueKindFact::Numeric(numeric) = &mut scaling.kind {
        numeric.domain = NumericDomain::Real;
    }
    let first = match request.outputs.requested.known_count() {
        Some(1) => combined,
        Some(_) => lower,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![
            first,
            upper,
            permutation(rows),
            permutation(columns),
            scaling,
        ],
        diagnostics,
    )
}
