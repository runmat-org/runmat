use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, usize::MAX, &mut diagnostics);
    let mut shape = ShapeFact::Scalar;
    let mut has_string = false;
    let mut has_cell = false;
    let mut unknown_kind = false;
    for (index, input) in request.arguments.iter().enumerate() {
        if invalid_text(input, CellTextPolicy::ScalarElements) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRCAT-TEXT",
                "strcat requires text operands",
                index,
            ));
        }
        has_string |= input.kind == ValueKindFact::String;
        has_cell |= matches!(input.kind, ValueKindFact::Cell(_));
        unknown_kind |= matches!(input.kind, ValueKindFact::Unknown)
            || invalid_text(input, CellTextPolicy::ScalarElements);
        match broadcast_shape(&shape, &text_shape(input)) {
            Ok(next) => shape = next,
            Err(error) => {
                diagnostics.push(error);
                shape = ShapeFact::Unknown;
            }
        }
    }
    let output = if unknown_kind {
        unknown()
    } else if has_string {
        host(ValueKindFact::String, shape)
    } else if has_cell {
        host(
            ValueKindFact::Cell(CellFact {
                element: Box::new(host(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![Some(1), None]),
                )),
                elements: Vec::new(),
                elements_complete: false,
            }),
            shape,
        )
    } else {
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![
                shape
                    .known_dims()
                    .and_then(|d| d.first().copied().flatten()),
                None,
            ]),
            StorageFact::Dense,
        )
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
            "RM-CATALOG-STRCAT-ARITY",
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
        let entry = &super::super::STRCAT_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments = vec![ValueFact::scalar(ValueKindFact::Logical); 1];
        assert!(!infer(&request, entry).diagnostics.is_empty());
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        request.arguments = vec![text, scalar.clone()];
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}
