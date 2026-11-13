use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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

#[test]
fn distribution_and_option_aliases_have_typed_identity() {
    assert_eq!(
        LassoglmDistribution::from_public_name("GAUSSIAN"),
        Some(LassoglmDistribution::Normal)
    );
    assert_eq!(
        LassoglmOption::from_public_name("Constant"),
        Some(LassoglmOption::Intercept)
    );
    assert_eq!(
        LassoglmOption::from_public_name("Tol_Fun"),
        Some(LassoglmOption::RelTol)
    );
    assert_eq!(
        LassoglmOption::Options.integer_role(),
        LassoglmOptionRole::Aggregate
    );
}

#[test]
fn known_distribution_and_requested_fitinfo_infer_double_matrix_and_struct() {
    let request = CallRequest {
        arguments: vec![
            numeric(6, 2),
            numeric(6, 1),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("binomial".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = inference::infer(&request, &LASSOGLM_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), None])
    );
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Struct(_)));
}

#[test]
fn predictor_shape_inference_accepts_singleton_tail_but_not_a_third_axis() {
    let mut request = CallRequest {
        arguments: vec![
            numeric(6, 3),
            numeric(6, 1),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("normal".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    request.arguments[0].shape = ShapeFact::from(vec![Some(6), Some(3), Some(1)]);
    let accepted = inference::infer(&request, &LASSOGLM_CATALOG_ENTRY);
    assert!(accepted.diagnostics.is_empty());
    assert_eq!(
        accepted.outputs[0].shape,
        ShapeFact::from(vec![Some(3), None])
    );
    request.arguments[0].shape = ShapeFact::from(vec![Some(6), Some(3), Some(2)]);
    let rejected = inference::infer(&request, &LASSOGLM_CATALOG_ENTRY);
    assert!(!rejected.diagnostics.is_empty());
    request.arguments[0].shape = ShapeFact::from(vec![Some(6), Some(3), None]);
    let uncertain = inference::infer(&request, &LASSOGLM_CATALOG_ENTRY);
    assert!(uncertain.diagnostics.is_empty());
    assert_eq!(
        uncertain.outputs[0].shape,
        ShapeFact::from(vec![None, None])
    );
}

#[test]
fn dynamic_option_name_can_replace_visible_scalar_lambda() {
    let request = CallRequest {
        arguments: vec![
            numeric(6, 2),
            numeric(6, 1),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            numeric(1, 1),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            numeric(1, 3),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("normal".into()),
            LiteralValue::String("Lambda".into()),
            LiteralValue::Number(0.1),
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&request, &LASSOGLM_CATALOG_ENTRY);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), None])
    );
}
