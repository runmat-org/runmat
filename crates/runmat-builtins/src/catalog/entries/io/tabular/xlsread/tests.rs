use super::*;
use runmat_types::*;

#[test]
fn one_two_three_outputs_retain_native_classes_and_unknown_shapes() {
    let filename = ValueFact::scalar(ValueKindFact::String);
    for outputs in 0..=3 {
        let request = CallRequest {
            arguments: vec![filename.clone()],
            literals: LiteralContext::new(vec![LiteralValue::String("fixture.xlsx".into())]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        };
        let inferred = inference::infer(&request, &XLSREAD_CATALOG_ENTRY);
        assert_eq!(inferred.outputs.len(), outputs);
        if outputs > 0 {
            assert_eq!(
                inferred.outputs[0].kind,
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            );
        }
        if outputs > 1 {
            assert!(matches!(inferred.outputs[1].kind, ValueKindFact::Cell(_)));
        }
        if outputs > 2 {
            assert!(matches!(inferred.outputs[2].kind, ValueKindFact::Cell(_)));
        }
    }
}
