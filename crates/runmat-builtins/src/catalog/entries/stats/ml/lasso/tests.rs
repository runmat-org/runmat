use super::*;
use runmat_types::{
    CallRequest, DimensionFact, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn numeric(rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn call(lambda: Option<(ValueFact, LiteralValue)>, outputs: usize) -> CallRequest {
    let mut arguments = vec![numeric(6, 3), numeric(6, 1)];
    let mut literals = vec![LiteralValue::Unknown; 2];
    if let Some((value, literal)) = lambda {
        arguments.extend([
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            value,
        ]);
        literals.extend([LiteralValue::String("Lambda".into()), literal]);
    }
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn option_spellings_have_one_typed_decoder() {
    assert_eq!(
        LassoOption::from_public_name("Lambda_Ratio"),
        Some(LassoOption::LambdaRatio)
    );
    assert_eq!(
        LassoOption::from_public_name("DF-max"),
        Some(LassoOption::DfMax)
    );
    assert_eq!(LassoOption::from_public_name("unknown"), None);
    assert_eq!(
        LassoOption::Cv.integer_role(),
        LassoOptionRole::StructuralControl
    );
}

#[test]
fn output_count_and_lambda_dedup_uncertainty_are_sound() {
    let scalar = inference::infer(
        &call(Some((numeric(1, 1), LiteralValue::Number(0.1))), 3),
        &LASSO_CATALOG_ENTRY,
    );
    assert!(scalar.diagnostics.is_empty());
    assert_eq!(
        scalar.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert!(matches!(scalar.outputs[1].kind, ValueKindFact::Struct(_)));
    assert_eq!(scalar.outputs[2].shape, ShapeFact::Scalar);
    let vector = inference::infer(
        &call(Some((numeric(1, 3), LiteralValue::Unknown)), 1),
        &LASSO_CATALOG_ENTRY,
    );
    let ShapeFact::Shaped { dims } = &vector.outputs[0].shape else {
        panic!("coefficient matrix shape");
    };
    assert_eq!(dims[0], DimensionFact::Known(3));
    assert_eq!(dims[1], DimensionFact::Unknown);
}

#[test]
fn predictor_shape_inference_accepts_singleton_tail_but_not_a_third_axis() {
    let mut request = call(None, 1);
    request.arguments[0].shape = ShapeFact::from(vec![Some(6), Some(3), Some(1)]);
    let accepted = inference::infer(&request, &LASSO_CATALOG_ENTRY);
    assert!(accepted.diagnostics.is_empty());
    assert_eq!(
        accepted.outputs[0].shape,
        ShapeFact::from(vec![Some(3), None])
    );

    request.arguments[0].shape = ShapeFact::from(vec![Some(6), Some(3), Some(2)]);
    let rejected = inference::infer(&request, &LASSO_CATALOG_ENTRY);
    assert!(!rejected.diagnostics.is_empty());
    request.arguments[0].shape = ShapeFact::from(vec![Some(6), Some(3), None]);
    let uncertain = inference::infer(&request, &LASSO_CATALOG_ENTRY);
    assert!(uncertain.diagnostics.is_empty());
    assert_eq!(
        uncertain.outputs[0].shape,
        ShapeFact::from(vec![None, None])
    );
}

#[test]
fn dfmax_and_dynamic_option_names_do_not_prove_lambda_path_width() {
    let mut request = call(Some((numeric(1, 1), LiteralValue::Number(0.1))), 1);
    request.arguments.extend([
        ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        numeric(1, 1),
    ]);
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Lambda".into()),
        LiteralValue::Number(0.1),
        LiteralValue::String("DFmax".into()),
        LiteralValue::Number(1.0),
    ]);
    let filtered = inference::infer(&request, &LASSO_CATALOG_ENTRY);
    assert_eq!(
        filtered.outputs[0].shape,
        ShapeFact::from(vec![Some(3), None])
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Lambda".into()),
        LiteralValue::Number(0.1),
        LiteralValue::Unknown,
        LiteralValue::Number(1.0),
    ]);
    let dynamic = inference::infer(&request, &LASSO_CATALOG_ENTRY);
    assert_eq!(
        dynamic.outputs[0].shape,
        ShapeFact::from(vec![Some(3), None])
    );
}
