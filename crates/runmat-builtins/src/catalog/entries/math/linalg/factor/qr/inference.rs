use super::{QR_ERROR_INVALID_ARGUMENT, QR_ERROR_INVALID_INPUT};
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
    if request
        .arguments
        .first()
        .is_some_and(|a| a.storage == StorageFact::Sparse)
    {
        return sparse(request, entry);
    }
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

fn sparse(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = matrix_input(
        request,
        4,
        false,
        QR_ERROR_INVALID_ARGUMENT.code,
        QR_ERROR_INVALID_INPUT.code,
    );
    let matrix = &request.arguments[0];
    let (rows, columns) = dimensions(matrix).unwrap_or((None, None));
    if !matches!(
        &matrix.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                ..
            })
    ) {
        diagnostics.push(argument_error(
            QR_ERROR_INVALID_INPUT.code,
            "sparse QR requires single or double coefficients",
            0,
        ));
    }
    // A numeric scalar zero remains the legacy economy selector. Without its
    // value, a scalar second operand cannot be distinguished from scalar B.
    let second = request.arguments.get(1);
    let zero = request
        .literals
        .literal_args
        .get(1)
        .is_some_and(exact_zero_literal);
    let rhs = second.filter(|b| {
        !zero
            && (b.storage == StorageFact::Sparse
                || request.literals.literal_args.get(1).is_some_and(|literal| {
                    matches!(
                        literal,
                        LiteralValue::Number(_)
                            | LiteralValue::Real { .. }
                            | LiteralValue::Complex { .. }
                    )
                })
                || dimensions(b)
                    .is_some_and(|(r, c)| r.zip(c).is_some_and(|(r, c)| r != 1 || c != 1)))
    });
    let ambiguous = second.is_some_and(|b| {
        !zero
            && rhs.is_none()
            && matches!(&b.kind, ValueKindFact::Numeric(_) | ValueKindFact::Unknown)
    });
    let start = if rhs.is_some() { 2 } else { 1 };
    if request.arguments.len() - start > 2 {
        diagnostics.push(argument_error(
            QR_ERROR_INVALID_ARGUMENT.code,
            "at most two QR options are accepted",
            start + 2,
        ));
    }
    let mut economy = Some(false);
    let mut permutation = PermutationChoice::MatrixDefault;
    for index in start..request.arguments.len() {
        if request
            .literals
            .literal_args
            .get(index)
            .is_some_and(exact_zero_literal)
        {
            economy = Some(true);
            permutation = PermutationChoice::NumericZeroVector;
            continue;
        }
        match literal_factor_option(request, index) {
            Some(FactorOption::Economy) => economy = Some(true),
            Some(FactorOption::Matrix) => permutation = PermutationChoice::ExplicitMatrix(index),
            Some(FactorOption::Vector) => permutation = PermutationChoice::ExplicitVector(index),
            _ => {
                economy = None;
                permutation = PermutationChoice::Unknown;
            }
        }
    }
    if let PermutationChoice::ExplicitMatrix(index) | PermutationChoice::ExplicitVector(index) =
        permutation
    {
        if request
            .outputs
            .requested
            .known_count()
            .is_some_and(|n| n < 3)
        {
            diagnostics.push(argument_error(
                QR_ERROR_INVALID_ARGUMENT.code,
                "permutation output form requires three outputs",
                index,
            ));
        }
    }
    if let Some(b) = rhs {
        if dimensions(b).is_some_and(|(r, _)| rows.zip(r).is_some_and(|(a, b)| a != b)) {
            diagnostics.push(argument_error(
                QR_ERROR_INVALID_INPUT.code,
                "B row count must equal coefficient row count",
                1,
            ));
        }
        if !matches!(
            &b.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single | NumericClass::Double,
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                QR_ERROR_INVALID_INPUT.code,
                "B requires single or double matrix storage",
                1,
            ));
        }
        if request
            .outputs
            .requested
            .known_count()
            .is_some_and(|n| n < 2)
        {
            diagnostics.push(argument_error(
                QR_ERROR_INVALID_ARGUMENT.code,
                "sparse two-input QR requires two or three outputs",
                1,
            ));
        }
    }
    let height = match economy {
        Some(false) => rows,
        Some(true) => rows.zip(columns).map(|(m, n)| m.min(n)),
        None => None,
    };
    let mut q = factor_output(
        matrix,
        if ambiguous {
            ShapeFact::Ranked { rank: 2 }
        } else if let Some(b) = rhs {
            matrix_shape(height, dimensions(b).and_then(|(_, c)| c))
        } else {
            matrix_shape(rows, height)
        },
        false,
        false,
        true,
    );
    let mut r = factor_output(matrix, matrix_shape(height, columns), false, false, true);
    r.storage = StorageFact::Sparse;
    if ambiguous {
        q.kind = ValueKindFact::Unknown;
    }
    // Unlike the legacy dense converter, this native CSC path never erases
    // complex storage, including all-zero and empty outputs.
    if let ValueKindFact::Numeric(numeric) = &matrix.kind {
        r.kind = ValueKindFact::Numeric(*numeric);
        let domain = match rhs.map(|b| &b.kind) {
            Some(ValueKindFact::Numeric(b)) if b.domain == NumericDomain::Complex => {
                NumericDomain::Complex
            }
            Some(ValueKindFact::Unknown) => {
                q.kind = ValueKindFact::Unknown;
                numeric.domain
            }
            _ => numeric.domain,
        };
        if !matches!(rhs.map(|b| &b.kind), Some(ValueKindFact::Unknown)) && !ambiguous {
            q.kind = ValueKindFact::Numeric(NumericFact {
                class: numeric.class,
                domain,
            });
        }
    }
    let vector = matches!(
        permutation,
        PermutationChoice::NumericZeroVector | PermutationChoice::ExplicitVector(_)
    );
    let mut p = binary64_output(
        matrix,
        if matches!(permutation, PermutationChoice::Unknown) {
            ShapeFact::Ranked { rank: 2 }
        } else {
            matrix_shape(columns, if vector { Some(1) } else { columns })
        },
        true,
        true,
    );
    p.storage = if matches!(permutation, PermutationChoice::Unknown) {
        StorageFact::Unknown
    } else if vector {
        StorageFact::Dense
    } else {
        StorageFact::Sparse
    };
    let first = match request.outputs.requested.known_count() {
        Some(1) => r.clone(),
        Some(_) => q,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(entry, request, vec![first, r, p], diagnostics)
}
