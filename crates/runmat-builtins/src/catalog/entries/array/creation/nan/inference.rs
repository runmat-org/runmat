use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::constructor_facts::infer(
        request,
        entry,
        super::super::constructor_facts::FillClass::Floating,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    fn request(literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments: literals
                .iter()
                .map(|_| {
                    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }))
                })
                .collect(),
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        }
    }
    #[test]
    fn scalar_and_known_dimensions() {
        let entry = &super::super::NAN_CATALOG_ENTRY;
        assert_eq!(
            infer(&request(vec![]), entry).outputs[0]
                .shape
                .element_count(),
            Some(1)
        );
        let result = infer(
            &request(vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)]),
            entry,
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert!(result.diagnostics.is_empty());
    }
    #[test]
    fn unknown_size_vector_does_not_invent_dimensions() {
        let mut request = request(vec![LiteralValue::Unknown]);
        request.arguments[0].shape = ShapeFact::from(vec![Some(1), Some(2)]);
        let result = infer(&request, &super::super::NAN_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    }
    #[test]
    fn dimensions_are_values_and_invalid_dimensions_diagnose() {
        let entry = &super::super::NAN_CATALOG_ENTRY;
        let result = infer(
            &request(vec![LiteralValue::Vector(vec![
                LiteralValue::Number(3.0),
                LiteralValue::Number(4.0),
                LiteralValue::Number(1.0),
            ])]),
            entry,
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(4)])
        );
        assert!(!infer(&request(vec![LiteralValue::Number(0.5)]), entry)
            .diagnostics
            .is_empty());
        assert_eq!(
            infer(&request(vec![LiteralValue::Empty]), entry).outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
    }
    #[test]
    fn like_selects_representation_not_shape() {
        let mut request = request(vec![
            LiteralValue::String("like".into()),
            LiteralValue::Unknown,
        ]);
        request.arguments[1].shape = ShapeFact::from(vec![Some(5), Some(7)]);
        request.arguments[1].kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        });
        let result = infer(&request, &super::super::NAN_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape.element_count(), Some(1));
        assert_eq!(result.outputs[0].kind, request.arguments[1].kind);
    }
    #[test]
    fn requested_outputs_remain_checked() {
        let mut request = request(vec![]);
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer(&request, &super::super::NAN_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }

    #[test]
    fn unknown_dimensions_do_not_commit_to_dense_storage() {
        let request = request(vec![LiteralValue::Unknown]);
        let output = &infer(&request, &super::super::NAN_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.storage, StorageFact::Unknown);
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    #[test]
    fn resident_row_size_is_control_not_output_residency() {
        let mut request = request(vec![LiteralValue::Unknown]);
        request.arguments[0].shape = ShapeFact::from(vec![Some(1), Some(2)]);
        request.arguments[0].storage = StorageFact::Dense;
        request.arguments[0].residency = ResidencyFact::Device {
            provider: Some("owner".into()),
        };
        let output = &infer(&request, &super::super::NAN_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.residency, ResidencyFact::Host);
        assert_eq!(output.shape, ShapeFact::Unknown);
    }
    #[test]
    fn resident_like_is_dense_even_for_scalar_shape() {
        let mut request = request(vec![
            LiteralValue::String("like".into()),
            LiteralValue::Unknown,
        ]);
        request.arguments[1].residency = ResidencyFact::Device {
            provider: Some("owner".into()),
        };
        request.arguments[1].storage = StorageFact::Dense;
        let output = &infer(&request, &super::super::NAN_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.shape.element_count(), Some(1));
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.residency, request.arguments[1].residency);
    }
    #[test]
    fn sparse_like_preserves_storage_without_borrowing_shape() {
        let mut request = request(vec![
            LiteralValue::String("like".into()),
            LiteralValue::Unknown,
        ]);
        request.arguments[1].shape = ShapeFact::from(vec![Some(3), Some(4)]);
        request.arguments[1].storage = StorageFact::Sparse;
        let output = &infer(&request, &super::super::NAN_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.shape.element_count(), Some(1));
        assert_eq!(output.storage, StorageFact::Sparse);
    }
    #[test]
    fn unknown_option_does_not_fabricate_representation() {
        let mut request = request(vec![LiteralValue::Unknown]);
        request.arguments[0].kind = ValueKindFact::String;
        let output = &infer(&request, &super::super::NAN_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.storage, StorageFact::Unknown);
        assert_eq!(output.residency, ResidencyFact::Unknown);
        assert_eq!(output.kind, ValueKindFact::Unknown);
    }
    #[test]
    fn explicit_device_precision_is_provider_dependent() {
        let request = request(vec![LiteralValue::String("gpuArray".into())]);
        let output = &infer(&request, &super::super::NAN_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.kind, ValueKindFact::Unknown);
        assert_eq!(output.residency, ResidencyFact::Device { provider: None });
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.shape.element_count(), Some(1));
    }
    #[test]
    fn single_scalar_retains_typed_dense_storage() {
        let request = request(vec![LiteralValue::String("single".into())]);
        let output = &infer(&request, &super::super::NAN_CATALOG_ENTRY).outputs[0];
        assert_eq!(output.shape.element_count(), Some(1));
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.numeric().unwrap().class, NumericClass::Single);
    }
}
