use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 3, 3, &mut diagnostics);
    for (index, value) in request.arguments.iter().enumerate() {
        if invalid_text(value, CellTextPolicy::ScalarElements) && !(index == 1 && pattern(value)) {
            diagnostics.push(argument_error(
                "RM-CATALOG-REPLACE-TEXT",
                "replace requires text or a search pattern object",
                index,
            ));
        }
    }
    let invalid_mapping = request
        .arguments
        .get(1)
        .zip(request.arguments.get(2))
        .filter(|(old, new)| {
            !matches!(
                old.kind,
                ValueKindFact::Character | ValueKindFact::Object(_)
            ) && !matches!(new.kind, ValueKindFact::Character)
        })
        .and_then(|(old, new)| old.shape.element_count().zip(new.shape.element_count()))
        .is_some_and(|(a, b)| a == 0 || b == 0 || b != 1 && b != a);
    if invalid_mapping {
        diagnostics.push(argument_error(
            "RM-CATALOG-REPLACE-MAPPING",
            "replacement terms must be nonempty and scalar or match search term count",
            2,
        ));
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
            "RM-CATALOG-REPLACE-ARITY",
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
        let entry = &super::super::REPLACE_CATALOG_ENTRY;
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
