use super::*;
use runmat_types::*;

fn call(literals: Vec<LiteralValue>, facts: Vec<ValueFact>) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: facts,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &MISSING_CATALOG_ENTRY,
    )
}

fn integer() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }))
}

#[test]
#[cfg(target_pointer_width = "64")]
fn scalar_and_exact_integer_sizes() {
    let scalar = call(vec![], vec![]);
    assert_eq!(scalar.outputs[0].kind, ValueKindFact::String);
    assert_eq!(scalar.outputs[0].shape.element_count(), Some(1));
    let shaped = call(
        vec![
            LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::Integer {
                text: "0".into(),
                class: NumericClass::UInt64,
            },
        ],
        vec![integer(), integer()],
    );
    assert_eq!(
        shaped.outputs[0].shape,
        ShapeFact::from(vec![Some(9_007_199_254_740_993), Some(0)])
    );
    assert_eq!(shaped.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn vector_contents_not_vector_shape_control_output_extents() {
    let size_vector = ValueFact::proven(
        integer().kind,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    assert_eq!(
        call(vec![LiteralValue::Unknown], vec![size_vector.clone()]).outputs[0].shape,
        ShapeFact::Unknown
    );
    assert_eq!(
        call(
            vec![LiteralValue::Vector(vec![
                LiteralValue::Number(2.0),
                LiteralValue::Number(3.0),
            ])],
            vec![size_vector],
        )
        .outputs[0]
            .shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
}

#[test]
fn invalid_literal_sizes_and_excess_outputs_diagnose() {
    assert!(!call(vec![LiteralValue::Number(-1.0)], vec![integer()])
        .diagnostics
        .is_empty());
    let mut request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    assert!(!inference::infer(&request, &MISSING_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.arguments = vec![integer()];
    request.literals = LiteralContext::new(vec![LiteralValue::String("like".into())]);
    assert!(!inference::infer(&request, &MISSING_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
