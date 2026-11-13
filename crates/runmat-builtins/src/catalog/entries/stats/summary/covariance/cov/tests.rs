use super::*;
use runmat_types::*;

fn matrix(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn call(arguments: Vec<ValueFact>, requested: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(requested)),
        },
        &COV_CATALOG_ENTRY,
    )
}

#[test]
fn single_matrix_vector_pair_and_weight_shapes() {
    let result = call(vec![matrix(5, 3)], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
    assert_eq!(
        call(vec![matrix(1, 5)], 1).outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    assert_eq!(
        call(vec![matrix(5, 1), matrix(5, 1)], 1).outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        call(vec![matrix(5, 3), matrix(5, 1)], 1).outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
    assert_eq!(
        call(vec![matrix(5, 3), matrix(5, 3)], 1).outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
}

#[test]
fn incompatible_pairs_and_complex_inputs_are_diagnosed() {
    assert!(!call(vec![matrix(5, 3), matrix(5, 2)], 1)
        .diagnostics
        .is_empty());
    let mut input = matrix(4, 2);
    input.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    assert!(!call(vec![input], 1).diagnostics.is_empty());
    assert!(!call(vec![matrix(4, 2)], 2).diagnostics.is_empty());
}

#[test]
fn resident_output_class_and_placement_are_not_guessed() {
    let mut input = matrix(5, 2);
    input.residency = ResidencyFact::Device { provider: None };
    let result = call(vec![input], 1);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn output_precision_comes_from_observations_not_weights() {
    let mut single = matrix(5, 2);
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    assert_eq!(
        call(vec![single.clone()], 1).outputs[0]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Single
    );
    assert_eq!(
        call(vec![single.clone(), matrix(5, 2)], 1).outputs[0]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Double
    );
    assert_eq!(
        call(vec![single, matrix(5, 1)], 1).outputs[0]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Single
    );
}

#[test]
fn typed_row_option_and_wide_normalization_are_checked() {
    assert_eq!(
        CovRowsOption::parse("partialrows"),
        Some(CovRowsOption::PartialRows)
    );
    assert_eq!(CovRowsOption::parse("bogus"), None);
    let input = matrix(5, 2);
    let flag = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let result = inference::infer(
        &CallRequest {
            arguments: vec![input, flag],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Integer {
                    text: "18446744073709551615".into(),
                    class: NumericClass::UInt64,
                },
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &COV_CATALOG_ENTRY,
    );
    assert!(!result.diagnostics.is_empty());
}
