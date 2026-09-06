use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use crate::catalog::inference::{argument_error, finish_fixed};

pub(in crate::catalog::entries::io::repl_fs) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-ADDPATH-ARITY",
            "addpath requires at least one folder argument",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !super::super::search_path::input::supports(
            argument,
            super::super::search_path::input::Policy::AddFolders,
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ADDPATH-TEXT",
                "addpath expects text folders, text containers, or numeric character-code rows",
                index,
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
    fn requires_a_folder_and_returns_a_character_row() {
        let entry = crate::builtin_catalog_entry_by_name("addpath").expect("addpath entry");
        let missing = infer(&request(Vec::new()), entry);
        assert_eq!(missing.diagnostics.len(), 1);

        let valid = infer(
            &request(vec![ValueFact::scalar(ValueKindFact::String)]),
            entry,
        );
        assert!(valid.diagnostics.is_empty());
        assert_eq!(valid.outputs[0].kind, ValueKindFact::Character);
    }
}
