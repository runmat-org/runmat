use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 3, 3, &mut diagnostics);
    if request
        .arguments
        .first()
        .is_some_and(|v| invalid_text(v, CellTextPolicy::ArrayElements))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRREP-TEXT",
            "strrep requires text with textual cell elements",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().enumerate().skip(1) {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || matches!(value.kind, ValueKindFact::String)
            && value.shape.element_count().is_some_and(|n| n != 1)
            || matches!(value.kind, ValueKindFact::Character)
                && !crate::catalog::inference::possible_character_row(&value.shape)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRREP-PATTERN",
                "old and new must be scalar text",
                index,
            ));
        }
    }
    if let (Some(old), Some(new)) = (request.arguments.get(1), request.arguments.get(2)) {
        if matches!(
            (&old.kind, &new.kind),
            (ValueKindFact::String, ValueKindFact::Character)
                | (ValueKindFact::Character, ValueKindFact::String)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRREP-CLASS",
                "old and new must have the same text class",
                2,
            ));
        }
    }
    let output = request
        .arguments
        .first()
        .map_or_else(unknown, |v| preserved(v, true));
    finish_fixed(entry, request, output, diagnostics)
}
fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
fn arity(
    request: &CallRequest,
    min: usize,
    max: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !(min..=max).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRREP-ARITY",
            "unsupported input count",
            request.arguments.len().min(max),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_facts_reject_missing_and_nontext_inputs() {
        let entry = &super::super::STRREP_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments = vec![ValueFact::scalar(ValueKindFact::Logical); 3];
        assert!(!infer(&request, entry).diagnostics.is_empty());
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        request.arguments = vec![text, scalar.clone(), scalar.clone()];
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}
