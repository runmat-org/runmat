use crate::catalog::entries::io::repl_fs::search_path;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GENPATH-ARITY",
            "genpath accepts at most two arguments",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        if !search_path::input::supports(argument, search_path::input::Policy::Generate) {
            let role = if index == 0 { "folder" } else { "excludes" };
            diagnostics.push(argument_error(
                "RM-CATALOG-GENPATH-TEXT",
                format!("genpath expects {role} to be a character row or string scalar"),
                index,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        search_path::result::character_row(),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn accepts_scalar_text_and_returns_a_character_row() {
        let entry = crate::builtin_catalog_entry_by_name("genpath").expect("genpath entry");
        let result = infer(
            &request(vec![ValueFact::scalar(ValueKindFact::String)]),
            entry,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
    }

    #[test]
    fn rejects_excess_arguments_and_text_containers() {
        let entry = crate::builtin_catalog_entry_by_name("genpath").expect("genpath entry");
        let too_many = infer(
            &request(vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ]),
            entry,
        );
        assert!(!too_many.diagnostics.is_empty());

        let invalid = infer(
            &request(vec![ValueFact::scalar(ValueKindFact::Logical)]),
            entry,
        );
        assert!(!invalid.diagnostics.is_empty());
    }
}
