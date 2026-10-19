use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, shape: &[usize]) -> ValueFact {
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    );
    value.residency = ResidencyFact::Host;
    value
}
fn infer(arguments: Vec<ValueFact>, count: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &POL2CART_CATALOG_ENTRY,
    )
}

#[test]
fn polar_and_cylindrical_shapes_include_trailing_dimensions_and_elevation() {
    for (shapes, expected) in [
        (vec![vec![1, 3], vec![2, 1]], vec![2, 3]),
        (vec![vec![1, 1], vec![2, 1], vec![1, 3]], vec![2, 3]),
        (vec![vec![3, 4], vec![3, 4, 2]], vec![3, 4, 2]),
        (vec![vec![0, 1, 2], vec![1, 3, 1]], vec![0, 3, 2]),
    ] {
        let count = shapes.len();
        let result = infer(
            shapes
                .iter()
                .map(|shape| numeric(NumericClass::Double, shape))
                .collect(),
            count,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        for output in result.outputs {
            assert_eq!(output.shape, numeric(NumericClass::Double, &expected).shape);
            assert_eq!(output.residency, ResidencyFact::Unknown);
        }
    }
}

#[test]
fn xy_class_and_elevation_class_are_independent() {
    let result = infer(
        vec![
            numeric(NumericClass::Single, &[1, 2]),
            numeric(NumericClass::Double, &[1, 1]),
            numeric(NumericClass::Double, &[1, 1]),
        ],
        3,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        result.outputs[2].numeric().unwrap().class,
        NumericClass::Double
    );
    let result = infer(
        vec![
            numeric(NumericClass::UInt64, &[1, 2]),
            numeric(NumericClass::Double, &[1, 1]),
            numeric(NumericClass::Single, &[1, 1]),
        ],
        3,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[2].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn unknown_inputs_do_not_invent_class_shape_or_residency() {
    let result = infer(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            numeric(NumericClass::Double, &[1, 1]),
        ],
        1,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    let mut resident = numeric(NumericClass::Double, &[2, 1]);
    resident.residency = ResidencyFact::Device {
        provider: Some("owner".into()),
    };
    assert_eq!(
        infer(vec![resident, numeric(NumericClass::Double, &[1, 1])], 1).outputs[0].residency,
        ResidencyFact::Unknown
    );
}

#[test]
fn invalid_kind_shape_arity_and_output_requests_are_diagnosed() {
    let valid = numeric(NumericClass::Double, &[2, 1]);
    let mut complex = valid.clone();
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    let mut sparse = valid.clone();
    sparse.storage = StorageFact::Sparse;
    for arguments in [
        vec![valid.clone()],
        vec![valid.clone(), complex],
        vec![valid.clone(), sparse],
        vec![valid.clone(), ValueFact::scalar(ValueKindFact::Logical)],
        vec![valid.clone(), numeric(NumericClass::Double, &[3, 1])],
    ] {
        assert!(!infer(arguments, 1).diagnostics.is_empty());
    }
    assert!(!infer(vec![valid.clone(), valid.clone()], 3)
        .diagnostics
        .is_empty());
    assert!(infer(vec![valid.clone(), valid], 0).outputs.is_empty());
}
