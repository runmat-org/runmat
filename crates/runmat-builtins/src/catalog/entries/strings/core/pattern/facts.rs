//! Shared facts for the host pattern value and numeric repetition controls.

use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, InferenceDiagnostic, LiteralValue, NumericDomain, ObjectFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(crate) fn pattern_fact() -> ValueFact {
    let properties = BTreeMap::from([("Regex".into(), ValueFact::scalar(ValueKindFact::String))]);
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::PATTERN.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(crate) fn text_input_diagnostics(
    request: &CallRequest,
    name: &str,
) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            format!("RM-CATALOG-{name}-ARITY"),
            format!("{name} requires exactly one text input"),
            0,
        ));
    }
    if let Some(value) = request.arguments.first() {
        let shape_invalid = match value.kind {
            ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
            ValueKindFact::Character => value
                .shape
                .known_dims()
                .and_then(|dims| dims.first().copied().flatten())
                .is_some_and(|rows| rows > 1),
            ValueKindFact::Unknown => false,
            _ => true,
        };
        if shape_invalid || matches!(value.residency, ResidencyFact::Device { .. }) {
            diagnostics.push(argument_error(
                format!("RM-CATALOG-{name}-TEXT"),
                format!("{name} requires a host text scalar"),
                0,
            ));
        }
    }
    diagnostics
}

pub(crate) fn bounded_count_diagnostics(
    request: &CallRequest,
    name: &str,
    reject_resident: bool,
) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            format!("RM-CATALOG-{name}-ARITY"),
            format!("{name} accepts zero, one, or two count inputs"),
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        let kind_ok = matches!(&input.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
            || matches!(input.kind, ValueKindFact::Unknown);
        let literal = request.literals.literal_args.get(index);
        let infinity_max = index == 1 && literal.is_some_and(positive_infinity_literal);
        let known_invalid_literal = literal.is_some_and(|literal| {
            !infinity_max && matches!(literal_count(literal), Some(CountLiteral::Invalid))
        });
        if !kind_ok
            || input.shape.element_count().is_some_and(|count| count != 1)
            || (reject_resident && matches!(input.residency, ResidencyFact::Device { .. }))
            || known_invalid_literal
        {
            diagnostics.push(argument_error(
                format!("RM-CATALOG-{name}-COUNT"),
                format!("{name} requires a nonnegative integer-valued count"),
                index,
            ));
        }
    }
    if let [minimum, maximum, ..] = request.literals.literal_args.as_slice() {
        if let (Some(CountLiteral::Count(minimum)), Some(CountLiteral::Count(maximum))) =
            (literal_count(minimum), literal_count(maximum))
        {
            if minimum > maximum {
                diagnostics.push(argument_error(
                    format!("RM-CATALOG-{name}-RANGE"),
                    format!("{name} minimum count exceeds maximum"),
                    1,
                ));
            }
        }
    }
    diagnostics
}

enum CountLiteral {
    Count(usize),
    Invalid,
}

fn literal_count(value: &LiteralValue) -> Option<CountLiteral> {
    match value {
        LiteralValue::Integer { text, .. } => {
            if text.starts_with('-') {
                return Some(CountLiteral::Invalid);
            }
            Some(
                text.parse::<u128>()
                    .ok()
                    .and_then(|n| usize::try_from(n).ok())
                    .map_or(CountLiteral::Invalid, CountLiteral::Count),
            )
        }
        LiteralValue::Number(n) => Some(floating_count(*n)),
        LiteralValue::Real { text, .. } => text.parse::<f64>().ok().map(floating_count),
        LiteralValue::Bool(_) | LiteralValue::String(_) | LiteralValue::Character(_) => {
            Some(CountLiteral::Invalid)
        }
        _ => None,
    }
}

fn floating_count(value: f64) -> CountLiteral {
    if !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        CountLiteral::Invalid
    } else {
        CountLiteral::Count(value as usize)
    }
}

fn positive_infinity_literal(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Number(value) => value.is_infinite() && value.is_sign_positive(),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .is_some_and(|n| n.is_infinite() && n.is_sign_positive()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, NumericClass, NumericFact, OutputSelection, RequestedOutputCount,
    };

    #[test]
    fn bounded_count_inference_distinguishes_host_and_resident_routes() {
        let mut count = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        count.residency = ResidencyFact::Device { provider: None };
        let request = CallRequest {
            arguments: vec![count],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        assert!(bounded_count_diagnostics(&request, "lettersPattern", false).is_empty());
        assert!(bounded_count_diagnostics(&request, "wildcardPattern", false).is_empty());
        assert!(bounded_count_diagnostics(&request, "digitsPattern", true)
            .iter()
            .any(|d| d.code == "RM-CATALOG-digitsPattern-COUNT"));
    }

    #[test]
    fn pattern_fact_uses_shared_runtime_class_identity() {
        let fact = pattern_fact();
        let ValueKindFact::Object(object) = fact.kind else {
            panic!("pattern fact must be an object");
        };
        assert_eq!(
            object.runtime_class,
            Some(runmat_types::standard::PATTERN.owned())
        );
    }
}
