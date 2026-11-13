use super::*;
use runmat_types::{InferenceDiagnostic, LiteralValue, NumericDomain};

fn empty(fact: &ValueFact) -> bool {
    dimensions(fact).is_some_and(|(rows, cols)| rows == Some(0) || cols == Some(0))
}

pub(super) fn validate(
    request: &CallRequest,
    rhs: &ValueFact,
    order: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    for index in 2..=4 {
        let Some(control) = request.arguments.get(index) else {
            continue;
        };
        if empty(control) {
            continue;
        }
        if control.shape.is_definitely_nonscalar()
            || !matches!(
                control.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(runmat_types::NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            )
        {
            diagnostics.push(argument_error(
                "RM.GMRES.INVALID_INPUT",
                "restart, tol and maxit must be real scalars or empty",
                index,
            ));
        }
        let invalid = match request.literals.literal_args.get(index) {
            Some(literal @ LiteralValue::Integer { .. }) if index != 3 => {
                literal.integer_value_i128().is_some_and(|value| {
                    value <= 0
                        || usize::try_from(value).is_err()
                        || (index == 2 && order.is_some_and(|order| value > order as i128))
                })
            }
            Some(literal @ LiteralValue::Integer { .. }) => {
                literal.integer_value_i128().is_some_and(|value| value <= 0)
                    || literal.integer_is_exact_binary64() == Some(false)
            }
            Some(LiteralValue::Number(_) | LiteralValue::Real { .. }) => {
                request.literals.numeric_at(index).is_some_and(|value| {
                    !value.is_finite()
                        || value <= 0.0
                        || (index != 3
                            && (value.fract() != 0.0
                                || value >= usize::MAX as f64
                                || (index == 2 && order.is_some_and(|order| value > order as f64))))
                })
            }
            _ => false,
        };
        if invalid {
            diagnostics.push(argument_error(
                "RM.GMRES.INVALID_INPUT",
                "invalid positive tolerance or platform-sized iteration count",
                index,
            ));
        }
    }
    for index in 5..=7 {
        let Some(value) = request.arguments.get(index) else {
            continue;
        };
        if empty(value) {
            continue;
        }
        if index != 7 && matches!(value.kind, ValueKindFact::Callable(_)) {
            continue;
        }
        if !matches!(
            value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || value
            .numeric()
            .zip(rhs.numeric())
            .is_some_and(|(left, right)| left != right)
        {
            diagnostics.push(argument_error(
                "RM.GMRES.UNSUPPORTED_INPUT",
                "preconditioners and x0 must match b's native floating class",
                index,
            ));
        }
        if let Some((rows, cols)) = dimensions(value) {
            if (index == 7 && cols.is_some_and(|cols| cols != 1))
                || (index != 7 && rows.zip(cols).is_some_and(|(rows, cols)| rows != cols))
                || order.zip(rows).is_some_and(|(order, rows)| order != rows)
            {
                diagnostics.push(argument_error(
                    "RM.GMRES.INVALID_INPUT",
                    "preconditioner or initial-vector geometry does not match the operator",
                    index,
                ));
            }
        }
    }
}
