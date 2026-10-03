use super::super::identities::FEA_STUDY_CLASS;
use super::FeaConstructorKind;
use runmat_types::{
    CertaintyFact, DimensionFact, DynamicReason, ObjectFact, ResidencyFact, ShapeFact,
    StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(crate) fn host_text(fact: &ValueFact) -> bool {
    if matches!(fact.residency, ResidencyFact::Device { .. }) {
        return false;
    }
    match fact.kind {
        // Unknown kind with shape [1, N] may still be a character row. Do not
        // reject it merely because a string scalar with that shape is invalid.
        ValueKindFact::Unknown => true,
        ValueKindFact::String => !matches!(fact.shape.element_count(), Some(count) if count != 1),
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

pub(super) fn possible_host_study(fact: &ValueFact) -> bool {
    !matches!(fact.residency, ResidencyFact::Device { .. })
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                !matches!(fact.shape.element_count(), Some(count) if count != 1)
                    && object
                        .runtime_class
                        .as_ref()
                        .is_none_or(|class| class.is(FEA_STUDY_CLASS))
            }
            _ => false,
        }
}

pub(super) fn object_result(kind: FeaConstructorKind, dynamic_document: bool) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: if dynamic_document {
                None
            } else {
                kind.class().map(StaticClassIdentity::owned)
            },
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    if dynamic_document {
        // The document may contain a Study or Sweep; path shape cannot tell.
        fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    fact
}
