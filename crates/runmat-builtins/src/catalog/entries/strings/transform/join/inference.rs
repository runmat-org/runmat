use crate::catalog::entries::strings::transform::text_facts::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, 1, 3, &mut diagnostics);
    if request
        .arguments
        .first()
        .is_some_and(|v| invalid_text(v, CellTextPolicy::ScalarElements))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-JOIN-TEXT",
            "join requires text input",
            0,
        ));
    }
    for (index, control) in request.arguments.iter().enumerate().skip(1) {
        if invalid_text(control, CellTextPolicy::ScalarElements)
            && !(real_control(control) && control.shape.element_count().is_none_or(|n| n == 1))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-JOIN-CONTROL",
                "join controls require text delimiters or a positive scalar dimension",
                index,
            ));
        }
    }
    let output = if let Some(input) = request.arguments.first() {
        let mut shape = text_shape(input);
        let dimension = request
            .arguments
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, v)| matches!(v.kind, ValueKindFact::Numeric(_)));
        let axis = match dimension {
            Some((index, _)) => request
                .literals
                .literal_args
                .get(index)
                .and_then(positive_dimension),
            None if request
                .arguments
                .iter()
                .skip(1)
                .any(|v| v.kind == ValueKindFact::Unknown) =>
            {
                None
            }
            None => shape.known_dims().and_then(|d| {
                d.iter()
                    .all(Option::is_some)
                    .then(|| d.iter().rposition(|n| *n != Some(1)).unwrap_or(1) + 1)
            }),
        };
        if let (Some(axis), Some(mut dims)) = (axis, shape.known_dims()) {
            if axis <= dims.len() {
                dims[axis - 1] = Some(1);
            }
            shape = ShapeFact::from(dims);
        } else {
            shape = ShapeFact::Unknown;
        }
        match input.kind {
            ValueKindFact::String | ValueKindFact::Character => host(ValueKindFact::String, shape),
            ValueKindFact::Cell(_) => host(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(host(
                        ValueKindFact::Character,
                        ShapeFact::from(vec![Some(1), None]),
                    )),
                    elements: Vec::new(),
                    elements_complete: false,
                }),
                shape,
            ),
            _ => unknown(),
        }
    } else {
        unknown()
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
            "RM-CATALOG-JOIN-ARITY",
            "unsupported input count",
            request.arguments.len().min(max),
        ));
    }
}

fn positive_dimension(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Number(v)
            if v.is_finite() && *v >= 1.0 && v.fract() == 0.0 && *v < usize::MAX as f64 =>
        {
            Some(*v as usize)
        }
        LiteralValue::Integer { .. } => literal
            .integer_value_i128()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(|v| positive_dimension(&LiteralValue::Number(v))),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_facts_reject_missing_and_nontext_inputs() {
        let entry = &super::super::JOIN_CATALOG_ENTRY;
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
            ShapeFact::from(vec![Some(2), Some(1)])
        );
        let mut renamed = *entry;
        renamed.identity = crate::BuiltinCatalogIdentity {
            name: "display-only-renamed",
        };
        assert_eq!(infer(&request, &renamed).outputs, result.outputs);
    }
}
