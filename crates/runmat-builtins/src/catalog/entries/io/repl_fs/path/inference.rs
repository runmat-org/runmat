use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use crate::catalog::inference::{argument_error, finish_fixed};

pub(in crate::catalog::entries::io::repl_fs) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PATH-ARITY",
            "path accepts at most two input arguments",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        if !super::super::search_path::input::supports(
            argument,
            super::super::search_path::input::Policy::PathReplacement,
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PATH-TEXT",
                "path expects a character row, string scalar, or numeric character-code row",
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
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn query_and_mutation_return_character_rows() {
        let entry = crate::builtin_catalog_entry_by_name("path").expect("path entry");
        for arguments in [
            Vec::new(),
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Character),
            ],
        ] {
            let result = infer(&request(arguments), entry);
            assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(1), None])
            );
            assert!(result.diagnostics.is_empty());
        }
    }

    #[test]
    fn admits_numeric_code_rows_and_rejects_known_invalid_forms() {
        let entry = crate::builtin_catalog_entry_by_name("path").expect("path entry");
        let numeric_row = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt16,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(4)]),
            StorageFact::Dense,
        );
        assert!(infer(&request(vec![numeric_row]), entry)
            .diagnostics
            .is_empty());

        let matrix = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        let result = infer(
            &request(vec![
                matrix,
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::scalar(ValueKindFact::String),
            ]),
            entry,
        );
        assert_eq!(result.diagnostics.len(), 3);
    }
}
