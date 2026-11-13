//! Shared fact admission for the two public mapped-object protocol methods.

use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, InferenceDiagnostic, ObjectFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn receiver_fact() -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::MEMMAPFILE_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.residency = ResidencyFact::Host;
    fact
}

fn validate_receiver_and_path(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(receiver) = request.arguments.first() {
        let wrong_class = match &receiver.kind {
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class != &super::MEMMAPFILE_CLASS.owned()),
            ValueKindFact::Unknown | ValueKindFact::Composite(_) => false,
            _ => true,
        };
        if wrong_class || receiver.shape.is_definitely_nonscalar() {
            diagnostics.push(argument_error(
                "RM-CATALOG-MEMMAPFILE-METHOD-RECEIVER",
                "receiver must be one memmapfile object",
                0,
            ));
        }
    }
    if let Some(path) = request.arguments.get(1) {
        if !matches!(
            &path.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown | ValueKindFact::Composite(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MEMMAPFILE-METHOD-PATH",
                "S must be a standard subscript structure",
                1,
            ));
        }
    }
}

pub(super) fn subsref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MEMMAPFILE-SUBSREF-ARITY",
            "subsref requires a receiver and standard subscript",
            0,
        ));
    }
    validate_receiver_and_path(request, &mut diagnostics);
    let mut result = ValueFact::unknown(DynamicReason::ExternalData);
    result.residency = ResidencyFact::Host;
    finish_fixed_outputs(entry, request, vec![result], diagnostics)
}

pub(super) fn subsasgn(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MEMMAPFILE-SUBSASGN-ARITY",
            "subsasgn requires a receiver, standard subscript, and rhs",
            0,
        ));
    }
    validate_receiver_and_path(request, &mut diagnostics);
    finish_fixed_outputs(entry, request, vec![receiver_fact()], diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount};

    fn call(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(vec![]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn method_facts_preserve_receiver_class_and_unknown_external_data() {
        let receiver = receiver_fact();
        let path = ValueFact::unknown(DynamicReason::RuntimeValue);
        let read = subsref(
            &call(vec![receiver.clone(), path.clone()]),
            &super::super::subsref::MEMMAPFILE_SUBSREF_CATALOG_ENTRY,
        );
        assert!(read.diagnostics.is_empty());
        assert!(matches!(&read.outputs[0].kind, ValueKindFact::Unknown));
        let write = subsasgn(
            &call(vec![
                receiver,
                path,
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ]),
            &super::super::subsasgn::MEMMAPFILE_SUBSASGN_CATALOG_ENTRY,
        );
        assert!(write.diagnostics.is_empty());
        let ValueKindFact::Object(object) = &write.outputs[0].kind else {
            panic!("mapped receiver")
        };
        assert_eq!(
            object
                .runtime_class
                .as_ref()
                .map(|class| class.display_name()),
            Some("memmapfile")
        );
    }

    #[test]
    fn definite_non_receiver_or_non_subscript_and_arity_reject() {
        let invalid = call(vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::String),
        ]);
        assert!(!subsref(
            &invalid,
            &super::super::subsref::MEMMAPFILE_SUBSREF_CATALOG_ENTRY
        )
        .diagnostics
        .is_empty());
        let unknown = call(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]);
        assert!(!subsasgn(
            &unknown,
            &super::super::subsasgn::MEMMAPFILE_SUBSASGN_CATALOG_ENTRY
        )
        .diagnostics
        .is_empty());

        let mut excess = call(vec![
            receiver_fact(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ]);
        excess.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!subsref(
            &excess,
            &super::super::subsref::MEMMAPFILE_SUBSREF_CATALOG_ENTRY
        )
        .diagnostics
        .is_empty());
    }
}
