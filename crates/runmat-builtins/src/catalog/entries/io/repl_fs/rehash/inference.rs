use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-REHASH-ARITY",
            "rehash accepts at most one mode",
            1,
        ));
    }
    if let Some(mode) = request.arguments.first() {
        if !super::super::text_input::scalar(mode) {
            diagnostics.push(argument_error(
                "RM-CATALOG-REHASH-MODE",
                "rehash mode must be a host string scalar or character row",
                0,
            ));
        }
        if let Some(mode) = request.literals.literal_string_at(0) {
            if super::RehashMode::parse(&mode).is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REHASH-UNKNOWN-MODE",
                    "rehash mode must be path, toolbox, or toolboxcache",
                    0,
                ));
            }
        }
    }
    let status = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    finish_fixed_outputs(entry, request, vec![status], diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount};

    #[test]
    fn known_mode_and_output_facts_match_runtime() {
        let entry = &super::super::REHASH_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String("toolboxcache".into())]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let known = infer(&request, entry);
        assert!(known.diagnostics.is_empty());
        assert_eq!(known.outputs.len(), 1);
        let mixed_case = CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::String("ToolBoxCache".into())]),
            ..request.clone()
        };
        assert!(infer(&mixed_case, entry).diagnostics.is_empty());
        let invalid = CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::String("bogus".into())]),
            ..request
        };
        assert!(infer(&invalid, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-REHASH-UNKNOWN-MODE"));
    }
}
