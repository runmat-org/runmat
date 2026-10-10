use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, LiteralValue, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ISSPACE-ARITY",
            "isspace requires exactly one input",
            0,
        ));
    }
    let output = match request.arguments.first() {
        Some(input) => match &input.kind {
            ValueKindFact::Character => logical_with_shape(input.shape.clone()),
            ValueKindFact::String
                if input.shape.element_count().is_some_and(|count| count != 1) =>
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ISSPACE-STRING",
                    "isspace string input must have exactly one element",
                    0,
                ));
                dynamic_logical()
            }
            ValueKindFact::String
                if input.storage == StorageFact::Dense
                    && input.shape.element_count() == Some(1) =>
            {
                // A one-element string array is unwrapped by the runtime before classification.
                logical_with_shape(ShapeFact::Unknown)
            }
            ValueKindFact::String => {
                let shape = match request.literals.literal_args.first() {
                    Some(LiteralValue::String(text)) => {
                        ShapeFact::from(vec![Some(1), Some(text.chars().count())])
                    }
                    _ => ShapeFact::Unknown,
                };
                logical_with_shape(shape)
            }
            ValueKindFact::Unknown => dynamic_logical(),
            _ => ValueFact::scalar(ValueKindFact::Logical),
        },
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}
fn logical_with_shape(shape: ShapeFact) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(ValueKindFact::Logical, shape, storage);
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}
fn dynamic_logical() -> ValueFact {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Logical;
    output
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn nonscalar_string_is_invalid_but_numeric_returns_scalar_false() {
        let entry = &super::super::ISSPACE_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(2)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ISSPACE-STRING"));
        let mut nontext = request;
        nontext.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
        assert_eq!(
            infer(&nontext, entry).outputs[0],
            ValueFact::scalar(ValueKindFact::Logical)
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn one_element_dense_string_carrier_is_classified_with_dynamic_character_length() {
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(1)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::ISSPACE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert!(matches!(&result.outputs[0].kind, ValueKindFact::Logical));
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    }
}
