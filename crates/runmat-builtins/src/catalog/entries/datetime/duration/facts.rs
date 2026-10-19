use runmat_types::{
    DynamicReason, NumericClass, NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

/// A known non-duration class is not interchangeable with an unknown object.
pub(super) fn duration_class(value: &ValueFact) -> Option<bool> {
    match &value.kind {
        ValueKindFact::Object(object) => match &object.runtime_class {
            Some(class) => Some(class == &runmat_types::standard::DURATION.owned()),
            None if object.class.is_some() => Some(false),
            None => None,
        },
        ValueKindFact::Unknown => None,
        _ => Some(false),
    }
}

pub(super) fn duration(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DURATION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    );
    value.residency = ResidencyFact::Host;
    value
}

pub(super) fn datetime(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DATETIME.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    );
    value.residency = ResidencyFact::Host;
    value
}

pub(super) fn double(shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    value.residency = ResidencyFact::Host;
    value
}

pub(super) fn logical(shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let mut value = ValueFact::proven(ValueKindFact::Logical, shape, storage);
    value.residency = ResidencyFact::Host;
    value
}

pub(super) fn unknown_host(shape: ShapeFact) -> ValueFact {
    let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
    value.shape = shape;
    value.residency = ResidencyFact::Host;
    value
}

pub(super) fn real_numeric_or_logical(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Logical
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    )
}
