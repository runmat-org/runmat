use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::{argument_error, finish_fixed};

pub(in crate::catalog::inference) fn infer(
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
        if !is_path_text(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-PATH-TEXT",
                "path expects a character row, string scalar, or numeric character-code row",
                index,
            ));
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn is_path_text(argument: &ValueFact) -> bool {
    match &argument.kind {
        ValueKindFact::Character => is_known_row(&argument.shape),
        ValueKindFact::String => argument
            .shape
            .element_count()
            .is_none_or(|count| count == 1),
        ValueKindFact::Numeric(numeric) => {
            numeric.domain != NumericDomain::Complex
                && argument.storage != StorageFact::Sparse
                && is_known_row(&argument.shape)
        }
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn is_known_row(shape: &ShapeFact) -> bool {
    shape
        .known_dims()
        .is_none_or(|dims| dims.len() <= 2 && dims.first().is_none_or(|rows| *rows == Some(1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, NumericClass, NumericFact, OutputSelection, RequestedOutputCount,
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
