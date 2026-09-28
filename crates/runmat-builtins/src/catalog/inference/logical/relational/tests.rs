use super::*;
use runmat_types::{
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ResidencyFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn numeric(shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    fact.shape = shape;
    fact.storage = StorageFact::Dense;
    fact
}

#[test]
fn relational_inference_broadcasts_to_logical() {
    let entry = crate::builtin_catalog_entry_by_name("lt").expect("lt entry");
    let request = request(vec![
        numeric(ShapeFact::from(vec![Some(3), Some(1)])),
        numeric(ShapeFact::from(vec![Some(1), Some(4)])),
    ]);
    let inference = infer(&request, entry, RelationalOperator::LessThan);
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
}

#[test]
fn symbolic_relations_preserve_symbolic_results() {
    let entry = crate::builtin_catalog_entry_by_name("ge").expect("ge entry");
    let request = request(vec![
        ValueFact::scalar(ValueKindFact::Symbolic),
        numeric(ShapeFact::Scalar),
    ]);
    let inference = infer(&request, entry, RelationalOperator::GreaterThanOrEqual);
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs[0].kind, ValueKindFact::Symbolic);
    assert_eq!(inference.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn unknown_relations_do_not_exclude_symbolic_outputs() {
    let entry = crate::builtin_catalog_entry_by_name("eq").unwrap();
    let mut input = numeric(ShapeFact::Scalar);
    input.residency = ResidencyFact::Device {
        provider: Some("test".into()),
    };
    let result = infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue), input]),
        entry,
        RelationalOperator::Equal,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn character_string_comparison_uses_whole_text_not_character_shape() {
    let entry = crate::builtin_catalog_entry_by_name("eq").unwrap();
    let chars = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(5)]),
        StorageFact::Dense,
    );
    let strings = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer(
        &request(vec![chars, strings.clone()]),
        entry,
        RelationalOperator::Equal,
    );
    assert_eq!(result.outputs[0].shape, strings.shape);
}
