use super::*;
use runmat_types::*;

fn matrix(class: NumericClass, domain: NumericDomain, rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn call(arguments: Vec<ValueFact>, literals: LiteralContext, requested: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(requested)),
        },
        &CORRCOEF_CATALOG_ENTRY,
    )
}

#[test]
fn matrix_vector_and_paired_shapes_are_distinct() {
    let input_matrix = matrix(NumericClass::Double, NumericDomain::Real, 5, 3);
    let result = call(vec![input_matrix.clone()], LiteralContext::default(), 4);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 4);
    assert!(result
        .outputs
        .iter()
        .all(|output| output.shape == ShapeFact::from(vec![Some(3), Some(3)])));
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    let vector = matrix(NumericClass::Double, NumericDomain::Real, 1, 5);
    assert_eq!(
        call(vec![vector], LiteralContext::default(), 1).outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    assert_eq!(
        call(
            vec![input_matrix.clone(), input_matrix],
            LiteralContext::default(),
            1
        )
        .outputs[0]
            .shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    let scalar = matrix(NumericClass::Single, NumericDomain::Real, 1, 1);
    let paired_scalar = call(vec![scalar.clone(), scalar], LiteralContext::default(), 1);
    assert_eq!(
        paired_scalar.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    assert_eq!(
        paired_scalar.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn invalid_complex_outputs_and_paired_size_are_diagnosed() {
    let complex = matrix(NumericClass::Double, NumericDomain::Complex, 5, 2);
    assert!(!call(vec![complex], LiteralContext::default(), 2)
        .diagnostics
        .is_empty());
    let left = matrix(NumericClass::Double, NumericDomain::Real, 5, 2);
    let right = matrix(NumericClass::Double, NumericDomain::Real, 6, 2);
    assert!(!call(vec![left, right], LiteralContext::default(), 1)
        .diagnostics
        .is_empty());
}

#[test]
fn resident_output_choice_stays_dynamic() {
    let mut input = matrix(NumericClass::Single, NumericDomain::Real, 5, 2);
    input.residency = ResidencyFact::Device { provider: None };
    let result = call(vec![input], LiteralContext::default(), 1);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn paired_observation_precision_is_promoted_without_control_promotion() {
    let single = matrix(NumericClass::Single, NumericDomain::Real, 5, 2);
    let double = matrix(NumericClass::Double, NumericDomain::Real, 5, 2);
    assert_eq!(
        call(vec![single.clone()], LiteralContext::default(), 1).outputs[0]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Single
    );
    assert_eq!(
        call(vec![single.clone(), double], LiteralContext::default(), 1).outputs[0]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Double
    );
    let control = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let text = ValueFact::scalar(ValueKindFact::String);
    let result = call(
        vec![single, text, control],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("Alpha".into()),
            LiteralValue::Number(0.05),
        ]),
        1,
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn typed_option_boundary_rejects_unknown_values_and_duplicate_keys() {
    assert_eq!(
        CorrcoefOptionKey::parse("Rows"),
        Some(CorrcoefOptionKey::Rows)
    );
    assert_eq!(
        CorrcoefRowsOption::parse("pairwisecomplete"),
        Some(CorrcoefRowsOption::Pairwise)
    );
    assert_eq!(CorrcoefRowsOption::parse("unexpected"), None);
    let input = matrix(NumericClass::Double, NumericDomain::Real, 5, 2);
    let text = ValueFact::scalar(ValueKindFact::String);
    let malformed = call(
        vec![input.clone(), text.clone(), text.clone()],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("Rows".into()),
            LiteralValue::String("bad".into()),
        ]),
        1,
    );
    assert!(!malformed.diagnostics.is_empty());
    let duplicate = call(
        vec![input, text.clone(), text.clone(), text.clone(), text],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("Rows".into()),
            LiteralValue::String("all".into()),
            LiteralValue::String("Rows".into()),
            LiteralValue::String("complete".into()),
        ]),
        1,
    );
    assert!(!duplicate.diagnostics.is_empty());
}
