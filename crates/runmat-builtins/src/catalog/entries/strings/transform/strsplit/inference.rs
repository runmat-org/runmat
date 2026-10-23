use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, usize::MAX, &mut diagnostics);
    let output = match request.arguments.first() {
        Some(value) => match value.kind {
            ValueKindFact::String => {
                if value.shape.element_count().is_some_and(|n| n != 1) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-STRSPLIT-SCALAR",
                        "strsplit requires scalar text",
                        0,
                    ));
                }
                host(ValueKindFact::String, ShapeFact::from(vec![Some(1), None]))
            }
            ValueKindFact::Character => {
                if !crate::catalog::inference::possible_character_row(&value.shape) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-STRSPLIT-ROW",
                        "strsplit requires one character row",
                        0,
                    ));
                }
                host(
                    ValueKindFact::Cell(CellFact {
                        element: Box::new(host(
                            ValueKindFact::Character,
                            ShapeFact::from(vec![Some(1), None]),
                        )),
                        elements: Vec::new(),
                        elements_complete: false,
                    }),
                    ShapeFact::from(vec![Some(1), None]),
                )
            }
            ValueKindFact::Unknown => unknown(),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRSPLIT-TEXT",
                    "strsplit requires scalar text, not a cell or numeric container",
                    0,
                ));
                unknown()
            }
        },
        None => unknown(),
    };
    for (index, value) in request.arguments.iter().enumerate().skip(1) {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Logical
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRSPLIT-CONTROL",
                "strsplit controls must be delimiters or named text/logical options",
                index,
            ));
        }
    }
    crate::catalog::inference::finish_fixed_outputs(
        entry,
        request,
        vec![output.clone(), output],
        diagnostics,
    )
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
            "RM-CATALOG-STRSPLIT-ARITY",
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
        let entry = &super::super::STRSPLIT_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments = vec![ValueFact::scalar(ValueKindFact::Logical); 1];
        assert!(!infer(&request, entry).diagnostics.is_empty());
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let text = scalar.clone();
        request.arguments = vec![text];
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
}
