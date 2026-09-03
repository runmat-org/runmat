use super::integer_division::infer;
use crate::IDIVIDE_CATALOG_ENTRY;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

#[test]
fn integer_division_preserves_the_integer_operand_class_and_broadcast_shape() {
    let request = CallRequest {
        arguments: vec![
            numeric(
                NumericClass::UInt16,
                ShapeFact::from(vec![Some(2), Some(1)]),
            ),
            numeric(NumericClass::Double, ShapeFact::Scalar),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = infer(&request, &IDIVIDE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real
        })
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}
