use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, 1, &mut diagnostics);
    let output = match request.arguments.first() {
        Some(value) => {
            if invalid_text(value, CellTextPolicy::ScalarElements) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SPLITLINES-TEXT",
                    "splitlines requires text input",
                    0,
                ));
            }
            let scalar = match value.kind {
                ValueKindFact::String => value.shape.element_count().map(|n| n == 1),
                ValueKindFact::Character => value
                    .shape
                    .known_dims()
                    .and_then(|d| d.first().copied().flatten())
                    .map(|r| r <= 1),
                ValueKindFact::Cell(_) => Some(false),
                _ => None,
            };
            match scalar {
                Some(true) => host(ValueKindFact::String, ShapeFact::from(vec![None, Some(1)])),
                Some(false) => host(
                    ValueKindFact::Cell(CellFact {
                        element: Box::new(host(
                            ValueKindFact::String,
                            ShapeFact::from(vec![None, Some(1)]),
                        )),
                        elements: Vec::new(),
                        elements_complete: false,
                    }),
                    text_shape(value),
                ),
                None => unknown(),
            }
        }
        None => unknown(),
    };
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
            "RM-CATALOG-SPLITLINES-ARITY",
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
        let entry = &super::super::SPLITLINES_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments = vec![ValueFact::scalar(ValueKindFact::Logical); 1];
        assert!(!infer(&request, entry).diagnostics.is_empty());
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        request.arguments = vec![text];
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}
