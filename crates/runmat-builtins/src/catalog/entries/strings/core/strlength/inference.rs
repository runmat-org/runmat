use crate::{
    catalog::inference::{
        argument_error, finish_fixed, invalid_text_comparison_container, numeric_kind,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRLENGTH-ARITY",
            "strlength requires exactly one input",
            0,
        ));
    }
    let output = match request.arguments.first() {
        Some(input) => {
            if !matches!(
                &input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
            ) || invalid_text_comparison_container(input)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRLENGTH-TEXT",
                    "strlength requires text input with scalar text cell elements",
                    0,
                ));
            }
            let shape = match &input.kind {
                ValueKindFact::String => input.shape.clone(),
                ValueKindFact::Character => {
                    input.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                        ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
                    })
                }
                ValueKindFact::Cell(_) => input.shape.clone(),
                _ => ShapeFact::Unknown,
            };
            let storage = match shape.element_count() {
                Some(1) => StorageFact::Scalar,
                Some(_) => StorageFact::Dense,
                None => StorageFact::Unknown,
            };
            let mut value = ValueFact::proven(
                numeric_kind(NumericClass::Double, NumericDomain::Real),
                shape,
                storage,
            );
            if matches!(&input.kind, ValueKindFact::Unknown) || value.shape == ShapeFact::Unknown {
                value.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
            }
            value
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    fn request(input: ValueFact) -> CallRequest {
        CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        }
    }
    #[test]
    fn string_shape_is_preserved_but_character_width_is_not() {
        let entry = &super::super::STRLENGTH_CATALOG_ENTRY;
        let string = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
            StorageFact::Dense,
        );
        let output = infer(&request(string), entry);
        assert_eq!(
            output.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3), Some(4)])
        );
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(8)]),
            StorageFact::Dense,
        );
        let output = infer(&request(chars), entry);
        assert_eq!(
            output.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(1)])
        );
    }
    #[test]
    fn unknown_and_invalid_inputs_do_not_invent_a_result_shape() {
        let entry = &super::super::STRLENGTH_CATALOG_ENTRY;
        let unknown = infer(
            &request(ValueFact::unknown(DynamicReason::RuntimeValue)),
            entry,
        );
        assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
        let invalid = infer(&request(ValueFact::scalar(ValueKindFact::Logical)), entry);
        assert_eq!(invalid.outputs[0].shape, ShapeFact::Unknown);
        assert!(invalid
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STRLENGTH-TEXT"));
    }
}
