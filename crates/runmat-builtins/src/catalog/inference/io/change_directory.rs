use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ShapeFact, StorageFact, ValueFact, ValueKindFact};

use super::super::{argument_error, finish_fixed};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CD-ARITY",
            "cd accepts at most one input argument",
            1,
        ));
    }
    if let Some(input) = request.arguments.first() {
        let valid = match &input.kind {
            ValueKindFact::Character => input
                .shape
                .known_dims()
                .is_none_or(|dims| dims.first() == Some(&Some(1))),
            ValueKindFact::String => input.shape.element_count().is_none_or(|count| count == 1),
            ValueKindFact::Unknown => true,
            _ => false,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-CD-PATH",
                "cd expects a character row vector or string scalar",
                0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount};

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn types_query_and_mutation_results_as_character_rows() {
        let entry = crate::builtin_catalog_entry_by_name("cd").expect("cd entry");
        for arguments in [Vec::new(), vec![ValueFact::scalar(ValueKindFact::String)]] {
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
    fn rejects_known_invalid_path_shapes_and_excess_arguments() {
        let entry = crate::builtin_catalog_entry_by_name("cd").expect("cd entry");
        let character_matrix = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        assert_eq!(
            infer(&request(vec![character_matrix]), entry)
                .diagnostics
                .len(),
            1
        );
        assert_eq!(
            infer(
                &request(vec![
                    ValueFact::scalar(ValueKindFact::String),
                    ValueFact::scalar(ValueKindFact::String)
                ]),
                entry
            )
            .diagnostics
            .len(),
            1
        );
    }
}
