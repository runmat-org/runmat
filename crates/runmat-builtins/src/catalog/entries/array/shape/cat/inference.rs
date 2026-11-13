use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

const MAX_CAT_DIMENSION: usize = 1024;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CAT-ARITY",
            "cat needs a dimension and at least two arrays",
            request.arguments.len(),
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    let Some(control) = request.arguments.first() else {
        return finish_fixed(entry, request, output, diagnostics);
    };
    if !matches!(
        &control.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    ) || control.shape.is_definitely_nonscalar()
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CAT-DIMENSION",
            "cat dimension must be a positive real integer scalar",
            0,
        ));
    }
    let dimension = request
        .literals
        .literal_args
        .first()
        .and_then(literal_dimension);
    if matches!(dimension, Some(None)) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CAT-DIMENSION",
            "cat dimension is outside the supported positive range",
            0,
        ));
    }
    let like_index = if request.arguments.len() >= 5
        && request
            .literals
            .literal_string_at(request.arguments.len() - 2)
            .as_deref()
            == Some("like")
    {
        Some(request.arguments.len() - 2)
    } else {
        None
    };
    let operands = &request.arguments[1..like_index.unwrap_or(request.arguments.len())];
    if operands.len() >= 2 && diagnostics.is_empty() {
        if let Some(Some(dimension)) = dimension {
            if admits_exact_shape_projection(operands) && like_index.is_none() {
                let projected = infer_concatenate(dimension, operands);
                output = projected.fact;
                diagnostics.extend(projected.diagnostics);
            }
        }
        if matches!(&output.kind, ValueKindFact::Unknown)
            && operands
                .iter()
                .all(|operand| operand.kind == operands[0].kind)
        {
            output.kind = operands[0].kind.clone();
        }
        if like_index.is_none()
            && operands
                .iter()
                .all(|operand| operand.residency == ResidencyFact::Host)
        {
            output.residency = ResidencyFact::Host;
        }
    }
    if !diagnostics.is_empty() {
        output.shape = ShapeFact::Unknown;
    }
    output.alias = AliasFact::Unknown;
    output.view = ViewFact::Unknown;
    finish_fixed(entry, request, output, diagnostics)
}

// Some(None) is a known-invalid literal; None leaves the control dynamic.
fn literal_dimension(literal: &LiteralValue) -> Option<Option<usize>> {
    match literal {
        LiteralValue::Unknown => None,
        LiteralValue::Integer { .. } => Some(
            literal
                .integer_value_i128()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| (1..=MAX_CAT_DIMENSION).contains(value)),
        ),
        LiteralValue::Number(value) => Some(floating_dimension(*value)),
        LiteralValue::Real { text, .. } => {
            Some(text.parse::<f64>().ok().and_then(floating_dimension))
        }
        _ => Some(None),
    }
}

fn floating_dimension(value: f64) -> Option<usize> {
    (value.is_finite() && value.fract() == 0.0 && value >= 1.0 && value <= MAX_CAT_DIMENSION as f64)
        .then_some(value as usize)
}

fn admits_exact_shape_projection(operands: &[ValueFact]) -> bool {
    operands.iter().all(|operand| {
        operand.residency == ResidencyFact::Host
            && operand.shape.element_count().is_some_and(|count| count > 0)
            && matches!(operand.storage, StorageFact::Scalar | StorageFact::Dense)
    }) && (operands.iter().all(|operand| {
        matches!(
            &operand.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    }) || operands
        .iter()
        .all(|operand| matches!(&operand.kind, ValueKindFact::Character))
        || operands
            .iter()
            .all(|operand| matches!(&operand.kind, ValueKindFact::String))
        || operands
            .iter()
            .all(|operand| matches!(&operand.kind, ValueKindFact::Cell(_))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::array::shape::cat::CAT_CATALOG_ENTRY;

    #[test]
    fn known_dimension_projects_host_shape_and_invalid_dimension_refuses() {
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let request = |dimension| CallRequest {
            arguments: vec![
                ValueFact::scalar(numeric.clone()),
                ValueFact::proven(
                    numeric.clone(),
                    ShapeFact::from(vec![Some(2), Some(3)]),
                    StorageFact::Dense,
                ),
                ValueFact::proven(
                    numeric.clone(),
                    ShapeFact::from(vec![Some(2), Some(4)]),
                    StorageFact::Dense,
                ),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Number(dimension)]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let known = infer(&request(2.0), &CAT_CATALOG_ENTRY);
        assert!(known.diagnostics.is_empty());
        assert_eq!(
            known.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(7)])
        );
        let invalid = infer(&request(0.0), &CAT_CATALOG_ENTRY);
        assert!(!invalid.diagnostics.is_empty());
        assert_eq!(invalid.outputs[0].shape, ShapeFact::Unknown);
    }
}
