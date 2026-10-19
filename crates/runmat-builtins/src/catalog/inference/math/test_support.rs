use std::collections::BTreeMap;

use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StaticClassIdentity, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn numeric(
    class: NumericClass,
    domain: NumericDomain,
    shape: Vec<Option<usize>>,
) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

pub(super) fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

pub(super) fn object(class: StaticClassIdentity) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(class.owned()),
        properties: BTreeMap::from([(
            "Variables".into(),
            ValueFact::scalar(ValueKindFact::Unknown),
        )]),
        properties_complete: true,
        handle_semantics: None,
    }))
}
