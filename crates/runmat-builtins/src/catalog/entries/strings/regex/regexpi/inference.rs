use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::regexp::infer_for(request, entry, super::super::regexp::RegexBuiltin::Regexpi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::inference::numeric_kind;
    use runmat_types::{LiteralContext, OutputSelection};
    use runmat_types::{
        LiteralValue, NumericClass, NumericDomain, RequestedOutputCount, ShapeFact, StorageFact,
        ValueFact, ValueKindFact,
    };
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn scalar_indices_are_double_with_dynamic_match_count() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::REGEXPI_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].kind,
            numeric_kind(NumericClass::Double, NumericDomain::Real)
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn array_results_remain_containers_and_invalid_inputs_diagnose() {
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(1)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::REGEXPI_CATALOG_ENTRY);
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
        request.arguments[1] = ValueFact::scalar(ValueKindFact::Logical);
        assert!(infer(&request, &super::super::REGEXPI_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-REGEXPI-TEXT"));
        request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
        assert_eq!(
            infer(&request, &super::super::REGEXPI_CATALOG_ENTRY).outputs[0].kind,
            ValueKindFact::Unknown
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn selector_after_warning_operand_is_not_an_output() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String); 4],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("warnings".into()),
                LiteralValue::String("match".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        assert_eq!(
            infer(&request, &super::super::REGEXPI_CATALOG_ENTRY).outputs[0].kind,
            numeric_kind(NumericClass::Double, NumericDomain::Real)
        );
    }
}
