use crate::{
    catalog::inference::{
        argument_error, finish_fixed, invalid_text_comparison_container, text_comparison_result,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRCMP-ARITY",
            "strcmp requires exactly two inputs",
            request.arguments.len().min(1),
        ));
    }
    let direct_false = request.arguments.iter().take(2).any(|argument| {
        matches!(
            &argument.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
        )
    });
    if !direct_false {
        for (index, argument) in request.arguments.iter().take(2).enumerate() {
            if invalid_text_comparison_container(argument) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRCMP-TEXT",
                    "strcmp requires text containers or direct numeric inputs",
                    index,
                ));
            }
        }
    }
    let output =
        if let (Some(left), Some(right)) = (request.arguments.first(), request.arguments.get(1)) {
            let (output, shape_error) = text_comparison_result(left, right);
            if let Some(shape_error) = shape_error {
                diagnostics.push(shape_error);
            }
            output
        } else {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        CellFact, LiteralContext, OutputSelection, ShapeFact, StorageFact, ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn known_text_and_unknown_inputs_have_sound_shapes() {
        let entry = &super::super::STRCMP_CATALOG_ENTRY;
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let known = infer(
            &request(vec![text, ValueFact::scalar(ValueKindFact::String)]),
            entry,
        );
        assert_eq!(known.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(
            known.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert!(known.diagnostics.is_empty());

        let unknown = infer(
            &request(vec![
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::scalar(ValueKindFact::String),
            ]),
            entry,
        );
        assert_eq!(unknown.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn invalid_arity_and_container_have_diagnostics() {
        let entry = &super::super::STRCMP_CATALOG_ENTRY;
        let one = infer(
            &request(vec![ValueFact::scalar(ValueKindFact::String)]),
            entry,
        );
        assert!(one
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STRCMP-ARITY"));
        let invalid = infer(
            &request(vec![
                ValueFact::scalar(ValueKindFact::Struct(runmat_types::StructFact::scalar(
                    Default::default(),
                    true,
                ))),
                ValueFact::scalar(ValueKindFact::String),
            ]),
            entry,
        );
        assert!(invalid
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STRCMP-TEXT"));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn known_nonscalar_string_inside_cell_is_rejected() {
        let element = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let cell = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(element),
                elements: vec![],
                elements_complete: false,
            }),
            ShapeFact::Scalar,
            StorageFact::Dense,
        );
        let result = infer(
            &request(vec![cell, ValueFact::scalar(ValueKindFact::String)]),
            &super::super::STRCMP_CATALOG_ENTRY,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRCMP-TEXT"));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn direct_numeric_false_precedes_invalid_peer_container() {
        let numeric = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Double,
            domain: runmat_types::NumericDomain::Real,
        }));
        let invalid = ValueFact::scalar(ValueKindFact::Struct(runmat_types::StructFact::scalar(
            Default::default(),
            true,
        )));
        for arguments in [
            vec![numeric.clone(), invalid.clone()],
            vec![invalid.clone(), numeric.clone()],
        ] {
            let result = infer(&request(arguments), &super::super::STRCMP_CATALOG_ENTRY);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
        }
    }
}
