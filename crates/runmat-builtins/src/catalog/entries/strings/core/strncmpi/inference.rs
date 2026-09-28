use crate::{
    catalog::inference::{
        argument_error, finish_fixed, invalid_text_comparison_container, text_comparison_result,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericDomain, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRNCMPI-ARITY",
            "strncmpi requires exactly three inputs",
            request.arguments.len().min(2),
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        if invalid_text_comparison_container(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRNCMPI-TEXT",
                "strncmpi requires text containers or direct numeric inputs",
                index,
            ));
        }
    }
    let direct_false = request.arguments.iter().take(2).any(|argument| {
        matches!(
            &argument.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
        )
    });
    if !direct_false {
        if let Some(limit) = request.arguments.get(2) {
            let admissible = matches!(&limit.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real)
                || matches!(&limit.kind, ValueKindFact::Logical | ValueKindFact::Unknown);
            if !admissible || limit.shape.element_count().is_some_and(|count| count != 1) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRNCMPI-LIMIT",
                    "N must be a real integer-valued scalar",
                    2,
                ));
            }
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
    use runmat_types::{LiteralContext, OutputSelection, ShapeFact, StorageFact};
    #[test]
    fn string_shape_and_invalid_n_are_distinct() {
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![
                text.clone(),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::STRNCMPI_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape, text.shape);
        assert!(result.diagnostics.is_empty());
        let mut invalid = request;
        invalid.arguments[2] = ValueFact::scalar(ValueKindFact::String);
        let result = infer(&invalid, &super::super::STRNCMPI_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRNCMPI-LIMIT"));
    }
}
