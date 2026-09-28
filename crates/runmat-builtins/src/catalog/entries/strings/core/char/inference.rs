use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    for (index, input) in request.arguments.iter().enumerate() {
        let supported = matches!(input.kind, ValueKindFact::Numeric(n) if n.domain == NumericDomain::Real)
            || matches!(
                input.kind,
                ValueKindFact::Logical
                    | ValueKindFact::Character
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
                    | ValueKindFact::Unknown
            );
        if !supported {
            diagnostics.push(argument_error(
                "RM-CATALOG-CHAR-INPUT",
                "char requires real numeric or text input",
                index,
            ));
        }
        if input.shape.known_dims().is_some_and(|dims| dims.len() > 2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-CHAR-DIMENSIONS",
                "char input must be at most two-dimensional",
                index,
            ));
        }
    }
    let shape = match request.arguments.as_slice() {
        [] => ShapeFact::from(vec![Some(0), Some(0)]),
        [input]
            if matches!(
                input.kind,
                ValueKindFact::Character | ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) =>
        {
            // Numeric codes retain their matrix rows, but output width can change for invalid codes.
            if matches!(input.kind, ValueKindFact::Character) {
                input.shape.clone()
            } else {
                input.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                    ShapeFact::from(vec![dims.first().copied().flatten(), None])
                })
            }
        }
        [input] if matches!(input.kind, ValueKindFact::String | ValueKindFact::Cell(_)) => {
            ShapeFact::from(vec![input.shape.element_count(), None])
        }
        _ => ShapeFact::Unknown,
    };
    let mut output = ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense);
    if request
        .arguments
        .iter()
        .any(|input| matches!(input.kind, ValueKindFact::Unknown))
    {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, NumericClass, NumericFact, OutputSelection};
    fn call(inputs: Vec<ValueFact>) -> CallInference {
        infer(
            &CallRequest {
                arguments: inputs,
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::CHAR_CATALOG_ENTRY,
        )
    }
    #[test]
    fn no_input_and_matrix_conversion_have_sound_shapes() {
        assert_eq!(
            call(vec![]).outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        let numeric = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![numeric]).outputs[0].shape,
            ShapeFact::from(vec![Some(2), None])
        );
    }
    #[test]
    fn complex_numeric_is_diagnosed_and_unknown_is_dynamic() {
        let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }));
        assert!(!call(vec![complex]).diagnostics.is_empty());
        assert!(matches!(
            call(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]).outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
    }
}
