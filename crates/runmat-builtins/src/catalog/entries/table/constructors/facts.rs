use crate::catalog::inference::{argument_error, gathered_value_fact};
use runmat_types::*;
use std::collections::BTreeMap;
pub(super) fn object(
    class: StaticClassIdentity,
    properties: BTreeMap<String, ValueFact>,
    complete: bool,
) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: complete,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
pub(super) fn retained(value: &ValueFact) -> ValueFact {
    let mut v = gathered_value_fact(value);
    v.alias = AliasFact::Unknown;
    v
}
pub(super) fn text(request: &CallRequest, index: usize) -> Option<&str> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::Character(v) | LiteralValue::String(v) | LiteralValue::Keyword(v) => Some(v),
        _ => None,
    }
}
pub(super) fn names(request: &CallRequest, index: usize) -> Option<Vec<String>> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::Vector(values) => values
            .iter()
            .map(|v| match v {
                LiteralValue::String(v) | LiteralValue::Character(v) => Some(v.clone()),
                _ => None,
            })
            .collect(),
        LiteralValue::String(value) | LiteralValue::Character(value) => Some(vec![value.clone()]),
        _ => None,
    }
}
pub(super) fn row_count(value: &ValueFact) -> Option<usize> {
    if let ValueKindFact::Object(object) = &value.kind {
        if object.runtime_class.as_ref() == Some(&standard::CATEGORICAL.owned()) {
            return object.properties.get("Codes").and_then(row_count);
        }
        return None;
    }
    match value.shape.known_dims() {
        Some(dims) => dims.first().copied().flatten(),
        None if value.is_scalar() => Some(1),
        _ => None,
    }
}
pub(super) fn code_shape(value: &ValueFact) -> ShapeFact {
    match &value.kind {
        ValueKindFact::Unknown => ShapeFact::Unknown,
        ValueKindFact::Character => ShapeFact::from(vec![
            value
                .shape
                .known_dims()
                .and_then(|dims| dims.first().copied().flatten()),
            Some(1),
        ]),
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref() == Some(&standard::CATEGORICAL.owned()) =>
        {
            object
                .properties
                .get("Codes")
                .map(|v| v.shape.clone())
                .unwrap_or(ShapeFact::Unknown)
        }
        _ => value.shape.clone(),
    }
}
pub(super) fn categorical(shape: ShapeFact) -> ValueFact {
    object(
        standard::CATEGORICAL,
        BTreeMap::from([
            (
                "Codes".into(),
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    shape,
                    StorageFact::Dense,
                ),
            ),
            (
                "Categories".into(),
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                ),
            ),
            ("Ordinal".into(), ValueFact::scalar(ValueKindFact::Logical)),
            (
                "Protected".into(),
                ValueFact::scalar(ValueKindFact::Logical),
            ),
        ]),
        true,
    )
}
pub(super) fn arity(
    request: &CallRequest,
    min: usize,
    max: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request.arguments.len() < min || max.is_some_and(|n| request.arguments.len() > n) {
        diagnostics.push(argument_error(
            crate::TABLE_ERROR_INVALID_ARGUMENT.code,
            "unsupported constructor argument count",
            0,
        ));
    }
}
pub(super) fn scalar_element(value: &ValueFact) -> ValueFact {
    let mut result = match &value.kind {
        ValueKindFact::Numeric(n) if n.domain == NumericDomain::Complex => retained(value),
        ValueKindFact::Numeric(n) => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: n.class,
            domain: n.domain,
        })),
        ValueKindFact::String | ValueKindFact::Character => {
            ValueFact::scalar(ValueKindFact::String)
        }
        ValueKindFact::Logical => ValueFact::scalar(ValueKindFact::Logical),
        ValueKindFact::Cell(cell) => (*cell.element).clone(),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    result.alias = AliasFact::Unknown;
    result
}
pub(super) fn dictionary_shape(value: &ValueFact) -> ShapeFact {
    match &value.kind {
        ValueKindFact::Character => ShapeFact::from(vec![
            value
                .shape
                .known_dims()
                .and_then(|dims| dims.first().copied().flatten()),
            Some(1),
        ]),
        ValueKindFact::Numeric(n) if n.domain == NumericDomain::Complex => ShapeFact::Scalar,
        ValueKindFact::Object(_) | ValueKindFact::Struct(_) => ShapeFact::Scalar,
        ValueKindFact::Unknown => ShapeFact::Unknown,
        _ => value.shape.clone(),
    }
}
pub(super) fn cells(element: ValueFact, length: Option<usize>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: length == Some(0),
        }),
        ShapeFact::from(vec![Some(1), length]),
        StorageFact::Opaque,
    )
}
