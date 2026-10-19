use std::collections::BTreeMap;

use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, StructFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILEATTRIB-ARITY",
            "fileattrib expects a path and up to two attribute flags",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !super::super::text_input::scalar(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILEATTRIB-TEXT",
                "fileattrib paths and flags must be host string scalars or character rows",
                index,
            ));
        }
    }
    for index in 1..request.arguments.len() {
        if let Some(flag) = request.literals.literal_string_at(index) {
            match super::FileAttributeControl::parse(&flag) {
                Ok(_) => {}
                Err(super::FileAttributeControlError::ReadPermissionUnavailable) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FILEATTRIB-PERMISSION-UNAVAILABLE",
                        "read permission changes require filesystem provider permission support",
                        index,
                    ));
                }
                Err(super::FileAttributeControlError::UnsupportedFlag(_)) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FILEATTRIB-FLAG",
                        "unsupported fileattrib attribute flag",
                        index,
                    ));
                }
            }
        }
    }
    let status = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let attributes = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        false,
    )));
    finish_fixed_outputs(entry, request, vec![status, attributes], diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount};

    #[test]
    fn flags_and_output_facts_match_the_supported_runtime_subset() {
        let entry = &super::super::FILEATTRIB_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::String("sample.bin".into()),
                LiteralValue::String("-w".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        let valid = infer(&request, entry);
        assert!(valid.diagnostics.is_empty());
        assert_eq!(valid.outputs.len(), 2);
        assert!(matches!(valid.outputs[1].kind, ValueKindFact::Struct(_)));
        let unsupported = CallRequest {
            literals: LiteralContext::new(vec![
                LiteralValue::String("sample.bin".into()),
                LiteralValue::String("+r".into()),
            ]),
            ..request.clone()
        };
        assert!(infer(&unsupported, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-FILEATTRIB-PERMISSION-UNAVAILABLE"));
        let excessive_outputs = CallRequest {
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
            ..request
        };
        assert!(!infer(&excessive_outputs, entry).diagnostics.is_empty());
    }
}
