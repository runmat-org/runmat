use super::*;

pub(crate) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let shape = match request.arguments.as_slice() {
        [input] => {
            if !accepted(input) {
                diagnostics.push(invalid_input(
                    crate::NumericalReductionKind::CountNonzero,
                    "expected numeric, logical, or character data",
                ));
            }
            ShapeFact::Scalar
        }
        [input, dimension] => {
            if !accepted(input) {
                diagnostics.push(invalid_input(
                    crate::NumericalReductionKind::CountNonzero,
                    "expected numeric, logical, or character data",
                ));
            }
            if dimension
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
                || !matches!(
                    dimension.kind,
                    ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    }) | ValueKindFact::Logical
                        | ValueKindFact::Unknown
                )
            {
                diagnostics.push(invalid(entry, "dimension must be a positive real scalar"));
            }
            let axis = literal(request, 1).and_then(|value| match value {
                LiteralValue::Bool(true) => Some(1),
                LiteralValue::Bool(false) => None,
                _ => positive_axis(value),
            });
            if literal(request, 1).is_some_and(|value| !matches!(value, LiteralValue::Unknown))
                && axis.is_none()
            {
                diagnostics.push(invalid(
                    entry,
                    "dimension must be a positive integer scalar",
                ));
            }
            reduced_shape(&input.shape, axis)
        }
        _ => {
            diagnostics.push(invalid(
                entry,
                "expected one input and at most one dimension",
            ));
            ShapeFact::Unknown
        }
    };
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape.clone(),
        if shape.element_count() == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    );
    finish(request, entry, vec![output], diagnostics)
}

fn accepted(input: &ValueFact) -> bool {
    matches!(
        input.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    )
}
