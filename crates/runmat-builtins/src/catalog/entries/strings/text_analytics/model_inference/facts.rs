use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn object(class: Option<StaticClassIdentity>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: class.map(StaticClassIdentity::owned),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
pub(super) fn numeric(domain: NumericDomain, shape: ShapeFact, storage: StorageFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain,
        }),
        shape,
        storage,
    )
}
pub(super) fn text_shape(input: Option<&ValueFact>) -> ShapeFact {
    match input {
        Some(value) if matches!(value.kind, ValueKindFact::Character) => {
            let rows = value
                .shape
                .known_dims()
                .and_then(|d| d.first().copied().flatten());
            ShapeFact::from(vec![rows.map(|r| r.max(1)), Some(1)])
        }
        Some(value) if matches!(value.kind, ValueKindFact::String | ValueKindFact::Cell(_)) => {
            value.shape.clone()
        }
        _ => ShapeFact::Unknown,
    }
}
pub(super) fn rows(input: Option<&ValueFact>) -> Option<usize> {
    let input = input?;
    match input.kind {
        ValueKindFact::Object(_) | ValueKindFact::Unknown => None,
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_) => Some(1),
        _ => input.shape.known_dims()?.first().copied().flatten(),
    }
}
