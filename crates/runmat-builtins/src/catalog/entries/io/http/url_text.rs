//! Shared admission for the URL text family, not numeric or provider work.

use crate::catalog::inference::{argument_error, finish_fixed, known_invalid_text_scalar};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-URL-TEXT-ARITY",
            "URL text helpers require exactly one input",
            request.arguments.len().min(1),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || known_invalid_text_scalar(input)
            || matches!(
                input.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-URL-TEXT-INPUT",
                "URL text must be a host string scalar or character row",
                0,
            ));
        }
    }
    // Encoding changes byte width; decoding changes byte and character widths.
    // Neither inherits the input's container, residency, extent or alias.
    let output = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuiltinContractMaturity, URLDECODE_CATALOG_ENTRY, URLENCODE_CATALOG_ENTRY};

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn text_container_and_length_are_not_copied_to_the_character_result() {
        for entry in [&URLENCODE_CATALOG_ENTRY, &URLDECODE_CATALOG_ENTRY] {
            assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Complete);
            for input in [
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![Some(1), Some(0)]),
                    StorageFact::Dense,
                ),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ] {
                let result = infer(&request(vec![input]), entry);
                assert!(result.diagnostics.is_empty());
                assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
                assert_eq!(
                    result.outputs[0].shape,
                    ShapeFact::from(vec![Some(1), None])
                );
                assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
                assert_eq!(result.outputs[0].alias, AliasFact::Unique);
                assert!(result.capabilities.0.is_empty());
                assert_eq!(result.effects, entry.contract.effect_set());
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn known_nontext_geometry_residency_and_arity_are_rejected() {
        for entry in [&URLENCODE_CATALOG_ENTRY, &URLDECODE_CATALOG_ENTRY] {
            let mut resident = ValueFact::scalar(ValueKindFact::String);
            resident.residency = ResidencyFact::Device { provider: None };
            for input in [
                resident,
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(1)]),
                    StorageFact::Dense,
                ),
                ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![Some(0), Some(0)]),
                    StorageFact::Dense,
                ),
                ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![Some(2), Some(3)]),
                    StorageFact::Dense,
                ),
            ] {
                assert!(!infer(&request(vec![input]), entry).diagnostics.is_empty());
            }
            assert!(!infer(&request(vec![]), entry).diagnostics.is_empty());
            let mut multiple_outputs = request(vec![ValueFact::scalar(ValueKindFact::String)]);
            multiple_outputs.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
            assert!(!infer(&multiple_outputs, entry).diagnostics.is_empty());
        }
    }
}
