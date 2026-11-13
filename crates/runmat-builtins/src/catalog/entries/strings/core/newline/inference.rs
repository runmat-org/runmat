use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        vec![]
    } else {
        vec![argument_error(
            "RM-CATALOG-NEWLINE-ARITY",
            "newline takes no inputs",
            0,
        )]
    };
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::String),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[test]
    fn newline_is_a_scalar_string_and_rejects_arguments() {
        let entry = &super::super::NEWLINE_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let valid = infer(&request, entry);
        assert_eq!(valid.outputs[0], ValueFact::scalar(ValueKindFact::String));
        assert!(valid.diagnostics.is_empty());
        let invalid = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            ..request
        };
        assert!(infer(&invalid, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-NEWLINE-ARITY"));
    }
}
