use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-USERPATH-ARITY",
            "userpath accepts at most one input",
            1,
        ));
    }
    if let Some(argument) = request.arguments.first() {
        if !super::super::text_input::scalar(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-USERPATH-TEXT",
                "userpath requires host scalar text",
                0,
            ));
        }
        if request.literals.literal_string_at(0).as_deref() == Some("") {
            diagnostics.push(argument_error(
                "RM-CATALOG-USERPATH-EMPTY",
                "an empty custom path is not admitted; use clear",
                0,
            ));
        }
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
        LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ValueFact,
        ValueKindFact,
    };

    #[test]
    fn text_forms_and_known_invalid_values_match_runtime_admission() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String("clear".into())]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let entry = &super::super::USERPATH_CATALOG_ENTRY;
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        let bad = CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::String(String::new())]),
            ..request.clone()
        };
        assert!(infer(&bad, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-USERPATH-EMPTY"));
        let numeric = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
            ..request
        };
        assert!(infer(&numeric, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-USERPATH-TEXT"));
    }
}
