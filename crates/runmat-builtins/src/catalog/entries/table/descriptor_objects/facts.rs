use crate::catalog::inference::{argument_error, gathered_value_fact, known_invalid_text_scalar};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn object(class: StaticClassIdentity, fields: &[(&str, ValueFact)]) -> ValueFact {
    let properties: BTreeMap<_, _> = fields
        .iter()
        .map(|(key, value)| ((*key).into(), value.clone()))
        .collect();
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn retained(request: &CallRequest, index: usize) -> ValueFact {
    request
        .arguments
        .get(index)
        .map(gathered_value_fact)
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
}

pub(super) fn string() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}
pub(super) fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
pub(super) fn empty_cell() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Opaque,
    )
}
pub(super) fn empty_names() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(0), Some(1)]),
        StorageFact::Dense,
    )
}
pub(super) fn text(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if let Some(value) = request.arguments.get(index) {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || known_invalid_text_scalar(value)
        {
            diagnostics.push(argument_error(
                "RM-TABLE-DESCRIPTOR-TEXT",
                "expected a string scalar or character row vector",
                index,
            ));
        }
    }
}
pub(super) fn arity(
    request: &CallRequest,
    min: usize,
    max: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request.arguments.len() < min || max.is_some_and(|max| request.arguments.len() > max) {
        diagnostics.push(argument_error(
            "RM-TABLE-DESCRIPTOR-ARITY",
            "unsupported number of constructor arguments",
            0,
        ));
    }
}
pub(super) fn option_pairs(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    arity(request, 1, None, diagnostics);
    if !request.arguments.is_empty() && !(request.arguments.len() - 1).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-TABLE-DESCRIPTOR-PAIRS",
            "name-value options must be provided in pairs",
            1,
        ));
    }
}
pub(super) fn positive_double(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if let Some(value) = request.arguments.get(index) {
        let bad_class = !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real
                })
        );
        let bad_shape = value.shape.element_count().is_some_and(|n| n != 1);
        let bad_literal = request.literals.numeric_at(index).is_some_and(|n| {
            !n.is_finite() || n <= 0.0 || n.fract() != 0.0 || n >= usize::MAX as f64
        });
        if bad_class || bad_shape || bad_literal {
            diagnostics.push(argument_error(
                "RM-TABLE-DESCRIPTOR-POSITIVE-DOUBLE",
                "control must be a positive integer-valued double scalar",
                index,
            ));
        }
    }
}
pub(super) fn direct_integer(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Numeric(numeric) if numeric.class.integer_class().is_some())
}
pub(super) fn resident_numeric(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical
    ) && matches!(value.residency, ResidencyFact::Device { .. })
}
