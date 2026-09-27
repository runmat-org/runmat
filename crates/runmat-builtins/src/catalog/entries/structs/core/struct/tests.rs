use super::*;
use runmat_types::*;
fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    super::inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        },
        &STRUCT_CATALOG_ENTRY,
    )
}
fn number() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }))
}
#[test]
fn empty_template_is_empty_array_not_scalar() {
    let mut input = number();
    input.shape = ShapeFact::from(vec![Some(0), Some(2)]);
    let out = infer(vec![input], vec![]);
    assert_eq!(
        out.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    assert_eq!(out.outputs[0].storage, StorageFact::Dense);
}
#[test]
fn cell_fields_expand_and_preserve_position_facts() {
    let mut first = number();
    first.residency = ResidencyFact::Device {
        provider: Some("owner".into()),
    };
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: vec![first.clone(), number()],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let out = infer(
        vec![ValueFact::scalar(ValueKindFact::String), cell],
        vec![LiteralValue::String("data".into())],
    );
    assert_eq!(
        out.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    let ValueKindFact::Struct(fact) = &out.outputs[0].kind else {
        panic!("struct");
    };
    assert_eq!(fact.elements[0]["data"], first);
    assert!(fact.elements_complete);
    assert_eq!(out.outputs[0].residency, ResidencyFact::Host);
}
#[test]
fn duplicate_fields_are_diagnosed() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let out = infer(
        vec![text.clone(), number(), text, number()],
        vec![
            LiteralValue::String("a".into()),
            LiteralValue::Unknown,
            LiteralValue::String("a".into()),
        ],
    );
    assert!(!out.diagnostics.is_empty());
}
#[test]
fn unknown_field_value_can_expand_to_unknown_struct_shape() {
    let out = infer(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        vec![LiteralValue::String("a".into())],
    );
    assert_eq!(out.outputs[0].shape, ShapeFact::Unknown);
}
