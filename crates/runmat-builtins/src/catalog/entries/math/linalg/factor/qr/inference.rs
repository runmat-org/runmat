use super::{QR_ERROR_INVALID_ARGUMENT, QR_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    binary64_output, dimensions, factor_output, literal_factor_option, matrix_input, matrix_shape,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use crate::FactorOption;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericDomain, ResidencyFact,
    ShapeFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
enum PermutationChoice {
    MatrixDefault,
    NumericZeroVector,
    ExplicitMatrix(usize),
    ExplicitVector(usize),
    Unknown,
}

fn exact_zero_literal(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(value) => *value == 0.0,
        LiteralValue::Real { text, .. } => text.parse::<f64>().is_ok_and(|value| value == 0.0),
        LiteralValue::Integer { .. } => literal.integer_value_i128() == Some(0),
        LiteralValue::Bool(value) => !value,
        LiteralValue::Vector(values) if values.len() == 1 => exact_zero_literal(&values[0]),
        LiteralValue::Matrix(rows) if rows.len() == 1 && rows[0].len() == 1 => {
            exact_zero_literal(&rows[0][0])
        }
        _ => false,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = matrix_input(
        request,
        3,
        false,
        QR_ERROR_INVALID_ARGUMENT.code,
        QR_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let (rows, columns) = dimensions(&matrix).unwrap_or((None, None));
    let options_are_absent = request.arguments.len() == 1;
    let q_shape = if options_are_absent {
        matrix_shape(rows, rows)
    } else {
        ShapeFact::Ranked { rank: 2 }
    };
    let r_shape = if options_are_absent {
        matrix_shape(rows, columns)
    } else {
        ShapeFact::Ranked { rank: 2 }
    };
    let mut permutation = PermutationChoice::MatrixDefault;
    for index in 1..request.arguments.len() {
        if request
            .literals
            .literal_args
            .get(index)
            .is_some_and(exact_zero_literal)
        {
            permutation = PermutationChoice::NumericZeroVector;
            continue;
        }
        match literal_factor_option(request, index) {
            Some(FactorOption::Vector) => permutation = PermutationChoice::ExplicitVector(index),
            Some(FactorOption::Matrix) => permutation = PermutationChoice::ExplicitMatrix(index),
            Some(FactorOption::Economy) => {}
            Some(_) | None => {
                permutation = PermutationChoice::Unknown;
                break;
            }
        }
    }
    let permutation_shape = match permutation {
        PermutationChoice::NumericZeroVector | PermutationChoice::ExplicitVector(_) => {
            matrix_shape(columns, Some(1))
        }
        PermutationChoice::MatrixDefault | PermutationChoice::ExplicitMatrix(_) => {
            matrix_shape(columns, columns)
        }
        PermutationChoice::Unknown => ShapeFact::Ranked { rank: 2 },
    };
    if let PermutationChoice::ExplicitMatrix(index) | PermutationChoice::ExplicitVector(index) =
        permutation
    {
        if request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count < 3)
        {
            diagnostics.push(argument_error(
                QR_ERROR_INVALID_ARGUMENT.code,
                "permutation output form requires three outputs",
                index,
            ));
        }
    }
    let mut q = factor_output(&matrix, q_shape, false, false, false);
    let mut r = factor_output(&matrix, r_shape, false, false, false);
    if matrix.residency != ResidencyFact::Host {
        // A resident factor follows its admitted physical input class, which
        // can differ from a static source-language fact. Permutations remain
        // double through the separate binary64 output projection below.
        q.kind = ValueKindFact::Unknown;
        r.kind = ValueKindFact::Unknown;
    }
    if matches!(&matrix.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex)
    {
        // The host QR value converter may erase numerically zero imaginary
        // parts, while a provider may retain complex storage.
        q.kind = ValueKindFact::Unknown;
        r.kind = ValueKindFact::Unknown;
    }
    let permutation = binary64_output(&matrix, permutation_shape, true, false);
    let first = match request.outputs.requested.known_count() {
        Some(1) => r.clone(),
        Some(_) => q,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(entry, request, vec![first, r, permutation], diagnostics)
}
