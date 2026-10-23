use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 4, 4, &mut diagnostics);
    for index in [0, 3] {
        if request
            .arguments
            .get(index)
            .is_some_and(|v| invalid_text(v, CellTextPolicy::Recursive))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-REPLACE-BETWEEN-TEXT",
                "source and replacement must be text",
                index,
            ));
        }
    }
    for index in [1, 2] {
        if let Some(value) = request.arguments.get(index) {
            if !(real_control(value)
                || pattern(value)
                || !invalid_text(value, CellTextPolicy::ScalarElements))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REPLACE-BETWEEN-BOUNDARY",
                    "boundaries require real numeric positions, text, or patterns",
                    index,
                ));
            }
            if real_control(value)
                && request
                    .literals
                    .literal_args
                    .get(index)
                    .is_some_and(invalid_positive)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REPLACE-BETWEEN-POSITION",
                    "positions must be positive integers",
                    index,
                ));
            }
        }
    }
    if let Some(text) = request.arguments.first() {
        let shape = text_shape(text);
        for index in 1..=3 {
            if let Some(value) = request.arguments.get(index) {
                let control = if real_control(value) {
                    value.shape.clone()
                } else {
                    text_shape(value)
                };
                if control.element_count().is_some_and(|n| n != 1)
                    && matches!((shape.known_dims(),control.known_dims()),(Some(a),Some(b)) if a.iter().all(Option::is_some)&&b.iter().all(Option::is_some)&&a!=b)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-REPLACE-BETWEEN-SHAPE",
                        "controls must be scalar or match the source text shape",
                        index,
                    ));
                }
            }
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
            "RM-CATALOG-REPLACE_BETWEEN-ARITY",
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
fn invalid_positive(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Vector(values) => values.iter().any(invalid_positive),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(invalid_positive),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|value| !value.is_finite() || value < 1.0 || value.fract() != 0.0),
        _ => {
            invalid_nonnegative(literal)
                || matches!(literal,LiteralValue::Number(v) if *v==0.0)
                || literal.integer_value_i128() == Some(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_facts_reject_missing_and_nontext_inputs() {
        let entry = &super::super::REPLACE_BETWEEN_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments = vec![ValueFact::scalar(ValueKindFact::Logical); 4];
        assert!(!infer(&request, entry).diagnostics.is_empty());
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        request.arguments = vec![
            text,
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
            scalar.clone(),
        ];
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}
