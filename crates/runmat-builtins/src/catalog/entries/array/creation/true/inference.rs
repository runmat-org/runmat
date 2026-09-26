use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::constructor_facts::infer(
        request,
        entry,
        super::super::constructor_facts::FillClass::Logical,
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
        let entry = &super::super::TRUE_CATALOG_ENTRY;
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
        let result = infer(&request, &super::super::TRUE_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    }
    #[test]
    fn dimensions_are_values_and_invalid_dimensions_diagnose() {
        let entry = &super::super::TRUE_CATALOG_ENTRY;
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
        let result = infer(&request, &super::super::TRUE_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape.element_count(), Some(1));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    }
    #[test]
    fn requested_outputs_remain_checked() {
        let mut request = request(vec![]);
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer(&request, &super::super::TRUE_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}
