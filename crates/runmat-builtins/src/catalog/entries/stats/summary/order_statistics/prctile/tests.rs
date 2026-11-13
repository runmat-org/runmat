use super::*;
use runmat_types::*;

#[test]
fn explicit_dimension_and_request_count_preserve_other_axes() {
    let shape = ShapeFact::from(vec![Some(3), Some(4), Some(2)]);
    let output = inference::infer(
        &CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::UInt16,
                        domain: NumericDomain::Real,
                    }),
                    shape,
                    StorageFact::Dense,
                ),
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![Some(1), Some(3)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Number(3.0),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &PRCTILE_CATALOG_ENTRY,
    );
    assert!(output.diagnostics.is_empty(), "{:?}", output.diagnostics);
    assert_eq!(
        output.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4), Some(3)])
    );
    assert_eq!(
        output.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}
