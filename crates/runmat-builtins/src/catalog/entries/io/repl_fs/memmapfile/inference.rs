use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MEMMAPFILE-ARITY",
            "memmapfile requires a filename and complete Name/Value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(filename) = request.arguments.first() {
        if !super::super::text_input::scalar(filename) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MEMMAPFILE-FILENAME",
                "filename must be a string scalar or character row",
                0,
            ));
        }
    }
    for index in (1..request.arguments.len()).step_by(2) {
        if !super::super::text_input::scalar(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MEMMAPFILE-OPTION",
                "property names must be string scalars or character rows",
                index,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(index) {
            if !["Writable", "Offset", "Format", "Repeat"]
                .iter()
                .any(|allowed| name.eq_ignore_ascii_case(allowed))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MEMMAPFILE-UNKNOWN-OPTION",
                    "unsupported memmapfile property name",
                    index,
                ));
            }
        }
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::contract::MEMMAPFILE_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    output.residency = ResidencyFact::Host;
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        DynamicReason, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
    };

    #[test]
    fn infers_object_and_rejects_incomplete_pairs_and_known_unknown_option() {
        let entry = &super::super::MEMMAPFILE_CATALOG_ENTRY;
        let valid = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::String("sample.bin".into()),
                LiteralValue::String("Format".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let inferred = infer(&valid, entry);
        assert!(inferred.diagnostics.is_empty());
        let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
            panic!("memmapfile object fact");
        };
        assert_eq!(
            object
                .runtime_class
                .as_ref()
                .map(|class| class.display_name()),
            Some("memmapfile")
        );
        let incomplete = CallRequest {
            arguments: valid.arguments[..2].to_vec(),
            ..valid.clone()
        };
        assert!(!infer(&incomplete, entry).diagnostics.is_empty());
        let unknown = CallRequest {
            literals: LiteralContext::new(vec![
                LiteralValue::String("sample.bin".into()),
                LiteralValue::String("Bogus".into()),
            ]),
            ..valid.clone()
        };
        assert!(!infer(&unknown, entry).diagnostics.is_empty());
        let excess_outputs = CallRequest {
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            ..valid.clone()
        };
        assert!(!infer(&excess_outputs, entry).diagnostics.is_empty());
    }
}
