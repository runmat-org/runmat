use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESTOREDEFAULTPATH-ARITY",
            "restoredefaultpath accepts no inputs",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        super::super::search_path::result::character_row(),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
    };

    #[test]
    fn zero_input_result_and_invalid_input_are_explicit() {
        let entry = &super::super::RESTOREDEFAULTPATH_CATALOG_ENTRY;
        assert_eq!(
            entry.contract.compatibility,
            crate::BuiltinCompatibility::Matlab
        );
        assert_eq!(
            entry.extensions,
            &[super::super::RESTOREDEFAULTPATH_RETURN_EXTENSION]
        );
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        let invalid = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            ..request
        };
        assert!(infer(&invalid, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-RESTOREDEFAULTPATH-ARITY"));
    }
}
