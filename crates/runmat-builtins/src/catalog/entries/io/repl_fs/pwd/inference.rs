use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ShapeFact, StorageFact, ValueFact, ValueKindFact};

use crate::catalog::inference::{argument_error, finish_fixed};

pub(in crate::catalog::entries::io::repl_fs) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-CATALOG-PWD-ARITY",
            "pwd accepts no input arguments",
            0,
        )]
    };
    let output = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount};

    #[test]
    fn infers_character_row_and_rejects_arguments() {
        let entry = crate::builtin_catalog_entry_by_name("pwd").expect("pwd catalog entry");
        let valid = infer(
            &CallRequest {
                arguments: Vec::new(),
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            entry,
        );
        assert_eq!(valid.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(valid.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
        assert!(valid.diagnostics.is_empty());

        let invalid = infer(
            &CallRequest {
                arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            entry,
        );
        assert_eq!(invalid.diagnostics.len(), 1);
        assert_eq!(invalid.diagnostics[0].argument, Some(0));
    }
}
