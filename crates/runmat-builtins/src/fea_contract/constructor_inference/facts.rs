//! Conservative static value facts shared by every FEA inference rule. A
//! predicate returns `false` only when the fact proves the runtime would reject
//! the value; unknown kinds, classes, and shapes are always admitted.

use runmat_types::{
    CertaintyFact, DimensionFact, DynamicReason, NumericClass, NumericDomain, ObjectFact,
    ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(crate) fn resident(fact: &ValueFact) -> bool {
    matches!(fact.residency, ResidencyFact::Device { .. })
}

fn known_non_scalar(fact: &ValueFact) -> bool {
    matches!(fact.shape.element_count(), Some(count) if count != 1)
}

pub(crate) fn host_text(fact: &ValueFact) -> bool {
    if resident(fact) {
        return false;
    }
    match fact.kind {
        // Unknown kind with shape [1, N] may still be a character row. Do not
        // reject it merely because a string scalar with that shape is invalid.
        ValueKindFact::Unknown => true,
        ValueKindFact::String => !known_non_scalar(fact),
        ValueKindFact::Character => match &fact.shape {
            ShapeFact::Shaped { dims } => {
                !matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1)
            }
            ShapeFact::Scalar | ShapeFact::Ranked { rank: 2.. } | ShapeFact::Unknown => true,
            _ => false,
        },
        _ => false,
    }
}

/// A scalar host object whose class, when known, is one of `classes`.
pub(crate) fn possible_object(fact: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    !resident(fact)
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                !known_non_scalar(fact)
                    && object
                        .runtime_class
                        .as_ref()
                        .is_none_or(|class| classes.iter().any(|identity| class.is(*identity)))
            }
            _ => false,
        }
}

/// A scalar object, or a cell whose known elements are such objects.
pub(crate) fn possible_object_or_cell(fact: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    match &fact.kind {
        ValueKindFact::Cell(cell) => {
            !resident(fact)
                && possible_object(&cell.element, classes)
                && cell
                    .elements
                    .iter()
                    .all(|element| possible_object(element, classes))
        }
        _ => possible_object(fact, classes),
    }
}

/// Any host object or cell of objects; the runtime decodes the exact class.
pub(crate) fn possible_component_list(fact: &ValueFact) -> bool {
    !resident(fact)
        && matches!(
            fact.kind,
            ValueKindFact::Unknown | ValueKindFact::Object(_) | ValueKindFact::Cell(_)
        )
}

/// Whether the fact is proven to be an object of exactly `class`.
pub(crate) fn known_class(fact: &ValueFact, class: StaticClassIdentity) -> bool {
    matches!(&fact.kind, ValueKindFact::Object(object)
        if object.runtime_class.as_ref().is_some_and(|known| known.is(class)))
}

/// Logical scalar accepted by the runtime's strict logical decoder.
pub(crate) fn possible_logical_scalar(fact: &ValueFact) -> bool {
    !resident(fact)
        && !known_non_scalar(fact)
        && matches!(fact.kind, ValueKindFact::Logical | ValueKindFact::Unknown)
}

/// Real host numeric (not single precision) or logical scalar.
pub(crate) fn possible_exact_scalar(fact: &ValueFact) -> bool {
    !resident(fact)
        && !known_non_scalar(fact)
        && match &fact.kind {
            ValueKindFact::Unknown | ValueKindFact::Logical => true,
            ValueKindFact::Numeric(number) => {
                number.domain == NumericDomain::Real && number.class != NumericClass::Single
            }
            _ => false,
        }
}

pub(crate) fn object_fact(class: StaticClassIdentity) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

/// A host value object whose class depends on a runtime-only input class.
pub(crate) fn dynamic_object_fact() -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: None,
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    fact
}
