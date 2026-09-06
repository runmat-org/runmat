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
            "RM-CATALOG-RMPATH-ARITY",
            "rmpath requires at least one folder argument",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !super::super::search_path::input::supports(
            argument,
            super::super::search_path::input::Policy::RemoveFolders,
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RMPATH-TEXT",
                "rmpath expects character, string, or cell text containers",
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
        LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ValueFact, ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn rejects_numeric_inputs_without_weakening_the_result_fact() {
        let entry = crate::builtin_catalog_entry_by_name("rmpath").expect("rmpath entry");
        let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let invalid = infer(&request(vec![numeric]), entry);
        assert_eq!(invalid.diagnostics.len(), 1);
        assert_eq!(invalid.outputs[0].kind, ValueKindFact::Character);
    }
}
