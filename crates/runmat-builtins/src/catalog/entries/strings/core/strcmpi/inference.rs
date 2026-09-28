use crate::{
    catalog::inference::{
        argument_error, finish_fixed, invalid_text_comparison_container, text_comparison_result,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRCMPI-ARITY",
            "strcmpi requires exactly two inputs",
            request.arguments.len().min(1),
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if invalid_text_comparison_container(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRCMPI-TEXT",
                "strcmpi requires text containers or direct numeric inputs",
                index,
            ));
        }
    }
    let output =
        if let (Some(left), Some(right)) = (request.arguments.first(), request.arguments.get(1)) {
            let (output, shape_error) = text_comparison_result(left, right);
            if let Some(shape_error) = shape_error {
                diagnostics.push(shape_error);
            }
            output
        } else {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, ShapeFact, StorageFact, ValueKindFact};

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        }
    }

    #[test]
    fn character_rows_broadcast_against_string_columns() {
        let entry = &super::super::STRCMPI_CATALOG_ENTRY;
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(3), Some(8)]),
            StorageFact::Dense,
        );
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let inferred = infer(&request(vec![chars, strings]), entry);
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(2)])
        );
        assert!(inferred.diagnostics.is_empty());
    }

    #[test]
    fn bad_arity_is_not_silent() {
        let entry = &super::super::STRCMPI_CATALOG_ENTRY;
        let inferred = infer(&request(vec![]), entry);
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STRCMPI-ARITY"));
    }
}
