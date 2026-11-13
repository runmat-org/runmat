use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, LiteralValue, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPENTOLINE-ARITY",
            "opentoline expects two through four arguments",
            0,
        ));
    }
    for index in [0, 3] {
        if let Some(value) = request.arguments.get(index) {
            if !possible_text(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPENTOLINE-TEXT",
                    "opentoline filename and selection option must be scalar text",
                    index,
                ));
            }
        }
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|value| value.is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPENTOLINE-EMPTY",
            "opentoline filename must not be empty",
            0,
        ));
    }
    if request
        .literals
        .literal_string_at(3)
        .is_some_and(|value| value != "select")
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPENTOLINE-OPTION",
            "opentoline supports the select option",
            3,
        ));
    }
    for index in 1..request.arguments.len().min(3) {
        let fact = &request.arguments[index];
        let possible = match fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) => fact.shape.element_count().is_none_or(|n| n == 1),
            ValueKindFact::String | ValueKindFact::Character => possible_text(fact),
            _ => false,
        };
        if !possible
            || request
                .literals
                .literal_args
                .get(index)
                .is_some_and(invalid_position)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPENTOLINE-POSITION",
                "opentoline position must be a positive integer scalar",
                index,
            ));
        }
    }
    finish_fixed_outputs(entry, request, vec![], diagnostics)
}
fn possible_text(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown) || super::super::text_input::scalar(value)
}
fn invalid_position(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Number(value) => invalid_real(*value),
        LiteralValue::Real { .. } => {
            runmat_types::LiteralContext::numeric_from_literal(value).is_some_and(invalid_real)
        }
        LiteralValue::Integer { text, .. } => text
            .parse::<i128>()
            .is_ok_and(|n| n < 1 || n > usize::MAX as i128),
        LiteralValue::Character(text) | LiteralValue::String(text) => {
            text.trim().parse::<usize>().map_or(true, |n| n == 0)
        }
        _ => false,
    }
}
fn invalid_real(value: f64) -> bool {
    !value.is_finite()
        || value < 1.0
        || value.fract().abs() > f64::EPSILON
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    fn request(position: LiteralValue, outputs: usize) -> CallRequest {
        CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![LiteralValue::String("editor.m".into()), position]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        }
    }
    #[test]
    fn zero_outputs_retain_effects_and_invalid_positions_are_diagnosed() {
        let result = infer(
            &request(LiteralValue::Number(12.0), 0),
            &super::super::OPENTOLINE_CATALOG_ENTRY,
        );
        assert!(result.diagnostics.is_empty());
        assert!(result.outputs.is_empty());
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        for number in [0.0, -1.0, 1.5, f64::INFINITY] {
            assert!(!infer(
                &request(LiteralValue::Number(number), 0),
                &super::super::OPENTOLINE_CATALOG_ENTRY
            )
            .diagnostics
            .is_empty());
        }
        assert!(!infer(
            &request(LiteralValue::Number(1.0), 1),
            &super::super::OPENTOLINE_CATALOG_ENTRY
        )
        .diagnostics
        .is_empty());
    }
    #[test]
    fn native_integer_positions_do_not_round_through_double() {
        let literal = LiteralValue::Integer {
            text: "9007199254740993".into(),
            class: NumericClass::UInt64,
        };
        assert_eq!(invalid_position(&literal), usize::BITS < 64);
        assert!(invalid_position(&LiteralValue::Integer {
            text: "-1".into(),
            class: NumericClass::Int64
        }));
    }

    #[test]
    fn single_positions_use_represented_precision() {
        assert!(!invalid_position(&LiteralValue::Real {
            text: "1.00000001".into(),
            class: NumericClass::Single,
        }));
        assert!(invalid_position(&LiteralValue::Real {
            text: "1.00000001".into(),
            class: NumericClass::Double,
        }));
    }
}
