use super::*;
use runmat_types::*;

fn data(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn infer(options: Vec<(ValueFact, LiteralValue)>, requested: usize) -> CallInference {
    let mut arguments = vec![data(4, 2), data(3, 2)];
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    for (fact, literal) in options {
        arguments.push(fact);
        literals.push(literal);
    }
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(requested)),
        },
        &KNNSEARCH_CATALOG_ENTRY,
    )
}

fn name(value: &str) -> (ValueFact, LiteralValue) {
    (
        ValueFact::scalar(ValueKindFact::String),
        LiteralValue::String(value.to_owned()),
    )
}

fn number(value: f64) -> (ValueFact, LiteralValue) {
    (
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        LiteralValue::Number(value),
    )
}

#[test]
fn nearest_outputs_have_double_class_and_query_by_k_shape() {
    let default = infer(vec![], 2);
    assert!(default.diagnostics.is_empty(), "{:?}", default.diagnostics);
    for output in &default.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(3), Some(1)]));
        assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
    }
    let requested_two = infer(vec![name("K"), number(2.0)], 2);
    assert!(
        requested_two.diagnostics.is_empty(),
        "{:?}",
        requested_two.diagnostics
    );
    assert_eq!(
        requested_two.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
}

#[test]
fn ties_produce_cells_and_unknown_toggle_does_not_claim_numeric() {
    let tied = infer(
        vec![
            name("IncludeTies"),
            (
                ValueFact::scalar(ValueKindFact::Logical),
                LiteralValue::Bool(true),
            ),
        ],
        2,
    );
    assert!(tied.diagnostics.is_empty(), "{:?}", tied.diagnostics);
    assert!(matches!(tied.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(
        tied.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    let unknown = infer(
        vec![
            name("IncludeTies"),
            (
                ValueFact::scalar(ValueKindFact::Logical),
                LiteralValue::Unknown,
            ),
        ],
        1,
    );
    assert!(matches!(unknown.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn proven_bad_k_metric_and_method_are_diagnosed() {
    assert!(!infer(vec![name("K"), number(5.0)], 1)
        .diagnostics
        .is_empty());
    assert!(!infer(vec![name("Distance"), name("made-up")], 1)
        .diagnostics
        .is_empty());
    assert!(!infer(
        vec![
            name("Distance"),
            name("cosine"),
            name("NSMethod"),
            name("kdtree")
        ],
        1
    )
    .diagnostics
    .is_empty());
}

#[test]
fn resident_source_cannot_claim_a_proven_host_output() {
    let mut request = CallRequest {
        arguments: vec![data(3, 1), data(2, 1)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    request.arguments[0].residency = ResidencyFact::Device { provider: None };
    let result = inference::infer(&request, &KNNSEARCH_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn floating_count_inference_matches_legacy_runtime_tolerance() {
    let result = infer(vec![name("K"), number(1.0 + 5.0e-13)], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert!(!infer(vec![name("K"), number(1.0 + 5.0e-10)], 1)
        .diagnostics
        .is_empty());
}

#[test]
fn logical_observation_fact_matches_runtime_admission() {
    let mut request = CallRequest {
        arguments: vec![data(3, 1), data(2, 1)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    request.arguments[0].kind = ValueKindFact::Logical;
    let result = inference::infer(&request, &KNNSEARCH_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}

#[test]
fn rank_one_observations_have_one_feature_in_static_geometry() {
    let mut request = CallRequest {
        arguments: vec![data(4, 1), data(3, 1)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    request.arguments[0].shape = ShapeFact::from(vec![Some(4)]);
    request.arguments[1].shape = ShapeFact::from(vec![Some(3)]);
    let shaped = inference::infer(&request, &KNNSEARCH_CATALOG_ENTRY);
    assert!(shaped.diagnostics.is_empty(), "{:?}", shaped.diagnostics);
    assert_eq!(
        shaped.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );

    request.arguments[1].shape = ShapeFact::Ranked { rank: 1 };
    let ranked = inference::infer(&request, &KNNSEARCH_CATALOG_ENTRY);
    assert!(ranked.diagnostics.is_empty(), "{:?}", ranked.diagnostics);
    assert_eq!(
        ranked.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
}
