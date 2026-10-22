use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, 4, &mut diagnostics);
    let input = request.arguments.first();
    if input.is_some_and(|v| invalid_text(v, CellTextPolicy::ScalarElements)) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PAD-TEXT",
            "pad requires text with scalar text cell elements",
            0,
        ));
    }
    for (index, control) in request.arguments.iter().enumerate().skip(1) {
        let numeric = index == 1 && real_control(control);
        if !(numeric
            || matches!(
                control.kind,
                ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
            ))
            || (numeric && control.shape.element_count().is_some_and(|n| n != 1))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PAD-CONTROL",
                "pad controls require a scalar real target or text direction/fill character",
                index,
            ));
        }
    }
    if request
        .literals
        .literal_args
        .get(1)
        .is_some_and(invalid_nonnegative)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PAD-LENGTH",
            "target must be a nonnegative integer",
            1,
        ));
    }
    let output = input.map_or_else(unknown, |v| preserved(v, true));
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
            "RM-CATALOG-PAD-ARITY",
            "unsupported input count",
            request.arguments.len().min(max),
        ));
    }
}
fn invalid_nonnegative(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(v) => !v.is_finite() || *v < 0.0 || v.fract() != 0.0,
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|v| !v.is_finite() || v < 0.0 || v.fract() != 0.0),
        LiteralValue::Integer { .. } => literal.integer_value_i128().is_some_and(|v| v < 0),
        LiteralValue::Vector(values) => values.iter().any(invalid_nonnegative),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(invalid_nonnegative),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_facts_reject_missing_and_nontext_inputs() {
        let entry = &super::super::PAD_CATALOG_ENTRY;
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
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}
