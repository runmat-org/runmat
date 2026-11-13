use crate::{
    catalog::inference::{argument_error, finish_fixed, literal_text},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-TEXT-BOUNDARY-ARITY",
            "textBoundary accepts zero or one type",
            0,
        ));
    }
    if let Some(value) = request.arguments.first() {
        let invalid = match &value.kind {
            ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
            ValueKindFact::Character => value
                .shape
                .known_dims()
                .and_then(|dims| dims.first().copied().flatten())
                .is_some_and(|rows| rows > 1),
            ValueKindFact::Unknown => false,
            _ => true,
        };
        if invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-TEXT-BOUNDARY-TYPE",
                "boundary type must be scalar text",
                0,
            ));
        }
        if let Some(option) = request.literals.literal_args.first().and_then(literal_text) {
            if !matches!(
                option.to_ascii_lowercase().as_str(),
                "either" | "start" | "end"
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TEXT-BOUNDARY-OPTION",
                    "boundary type must be either, start, or end",
                    0,
                ));
            }
        }
    }
    finish_fixed(
        entry,
        request,
        super::super::pattern::facts::pattern_fact(),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection, ValueFact};
    #[test]
    fn known_invalid_boundary_option_is_diagnosed() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String("middle".into())]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::TEXT_BOUNDARY_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-TEXT-BOUNDARY-OPTION"));
    }
}
