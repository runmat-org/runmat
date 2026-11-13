use crate::catalog::inference::{
    argument_error, default_double_scalar, finish_fixed, known_invalid_text_scalar, literal_text,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericDomain, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(runmat_types::InferenceDiagnostic::error(
            "RM-CATALOG-PAUSE-ARITY",
            "pause accepts at most one input",
        ));
    }
    let output = if let Some(argument) = request.arguments.first() {
        if let Some(text) = request.literals.literal_args.first().and_then(literal_text) {
            match text.trim().to_ascii_lowercase().as_str() {
                "on" | "off" | "query" => state_text(),
                "" => empty_result(),
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-PAUSE-COMMAND",
                        "pause command must be on, off, or query",
                        0,
                    ));
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
            }
        } else {
            match argument.kind {
                ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-PAUSE-DURATION",
                        "pause duration must be real",
                        0,
                    ));
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
                ValueKindFact::Numeric(_) | ValueKindFact::Logical => {
                    if argument
                        .shape
                        .element_count()
                        .is_some_and(|count| count > 1)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PAUSE-DURATION",
                            "pause duration must be scalar or empty",
                            0,
                        ));
                    }
                    if request
                        .literals
                        .literal_args
                        .first()
                        .is_some_and(invalid_duration_literal)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PAUSE-DURATION",
                            "pause duration must be nonnegative and not NaN",
                            0,
                        ));
                    }
                    empty_result()
                }
                ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown => {
                    if known_invalid_text_scalar(argument) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PAUSE-COMMAND",
                            "pause command must be scalar text",
                            0,
                        ));
                    }
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-PAUSE-INPUT",
                        "pause requires a duration or command",
                        0,
                    ));
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
            }
        }
    } else {
        empty_result()
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn invalid_duration_literal(literal: &LiteralValue) -> bool {
    let value = match literal {
        LiteralValue::Number(value) => Some(*value),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse::<f64>().ok()
        }
        _ => None,
    };
    value.is_some_and(|value| value.is_nan() || value < 0.0)
}

fn empty_result() -> ValueFact {
    let mut fact = default_double_scalar();
    fact.shape = ShapeFact::from(vec![Some(0), Some(0)]);
    fact.storage = StorageFact::Dense;
    fact
}

fn state_text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Character)
}
