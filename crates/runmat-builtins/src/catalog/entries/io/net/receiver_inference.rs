use runmat_types::{StaticClassIdentity, ValueFact, ValueKindFact};

pub(super) fn may_be_receiver(fact: &ValueFact, class: StaticClassIdentity) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_none_or(|identity| identity.is(class)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DynamicReason, ObjectFact};
    use std::collections::BTreeMap;

    fn object(class: Option<StaticClassIdentity>) -> ValueFact {
        ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: class.map(StaticClassIdentity::owned),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(true),
        }))
    }

    #[test]
    fn receiver_class_rejects_only_definite_wrong_class() {
        assert!(may_be_receiver(
            &object(Some(super::super::TCPCLIENT_CLASS)),
            super::super::TCPCLIENT_CLASS
        ));
        assert!(!may_be_receiver(
            &object(Some(super::super::TCPSERVER_CLASS)),
            super::super::TCPCLIENT_CLASS
        ));
        assert!(may_be_receiver(
            &object(None),
            super::super::TCPCLIENT_CLASS
        ));
        assert!(may_be_receiver(
            &ValueFact::unknown(DynamicReason::RuntimeValue),
            super::super::TCPCLIENT_CLASS
        ));
        assert!(!may_be_receiver(
            &ValueFact::scalar(ValueKindFact::String),
            super::super::TCPCLIENT_CLASS
        ));
    }
}
