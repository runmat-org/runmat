use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, usize::MAX, &mut diagnostics);
    if request.arguments.first().is_some_and(|input| {
        invalid_text(input, CellTextPolicy::ScalarElements) || nonmatrix(&input.shape)
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SPLIT-TEXT",
            "split currently requires scalar or two-dimensional text containers",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().enumerate().skip(1) {
        if !(matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Logical
        ) || real_control(value))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SPLIT-CONTROL",
                "split requires text delimiters, a scalar dimension, or named controls",
                index,
            ));
        }
    }
    let output = host(ValueKindFact::String, output_shape(request));
    crate::catalog::inference::finish_fixed_outputs(
        entry,
        request,
        vec![output.clone(), output],
        diagnostics,
    )
}
fn output_shape(request: &CallRequest) -> ShapeFact {
    // Token counts depend on text contents. Only the surrounding geometry is
    // static; a dynamic explicit dimension cannot be guessed as the default.
    if request.arguments.get(2).is_some_and(real_control) {
        return ShapeFact::Unknown;
    }
    let Some(input) = request.arguments.first() else {
        return ShapeFact::Unknown;
    };
    let Some(dims) = text_shape(input).known_dims() else {
        return ShapeFact::Unknown;
    };
    let rows = dims.first().copied().flatten();
    let cols = dims.get(1).copied().flatten();
    if rows == Some(0) || cols == Some(0) {
        return ShapeFact::from(vec![rows, cols]);
    }
    match (rows, cols) {
        (Some(1), Some(1)) => ShapeFact::from(vec![None, Some(1)]),
        (_, Some(1)) => ShapeFact::from(vec![rows, None]),
        (_, Some(_)) => ShapeFact::from(vec![rows, cols, None]),
        _ => ShapeFact::Unknown,
    }
}
fn arity(
    request: &CallRequest,
    min: usize,
    max: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !(min..=max).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SPLIT-ARITY",
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
        let entry = &super::super::SPLIT_CATALOG_ENTRY;
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
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3), None])
        );
    }
}
