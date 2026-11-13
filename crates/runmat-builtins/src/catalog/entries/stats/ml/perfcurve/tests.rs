use super::*;
use runmat_types::*;

fn numeric(shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &PERFCURVE_CATALOG_ENTRY,
    )
}

#[test]
fn known_real_vectors_have_useful_double_and_cell_output_facts() {
    let inferred = call(
        vec![numeric(&[4, 1]), numeric(&[4, 1]), numeric(&[1, 1])],
        vec![LiteralValue::Unknown; 3],
        7,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs.len(), 7);
    for output in inferred.outputs.iter().take(6) {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    assert_eq!(
        inferred.outputs[4].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert!(matches!(inferred.outputs[6].kind, ValueKindFact::Cell(_)));
}

#[test]
fn invalid_or_ambiguous_required_inputs_do_not_prove_results() {
    for (labels, scores, positive) in [
        (numeric(&[4, 1]), numeric(&[3, 1]), numeric(&[1, 1])),
        (numeric(&[2, 2]), numeric(&[4, 1]), numeric(&[1, 1])),
        (numeric(&[4, 1]), numeric(&[2, 2]), numeric(&[1, 1])),
        (numeric(&[4, 1]), numeric(&[4, 1]), numeric(&[2, 1])),
    ] {
        let inferred = call(
            vec![labels, scores, positive],
            vec![LiteralValue::Unknown; 3],
            1,
        );
        assert!(!inferred.diagnostics.is_empty());
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    }
    let uncertain = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), None]),
        StorageFact::Dense,
    );
    let inferred = call(
        vec![numeric(&[4, 1]), uncertain, numeric(&[1, 1])],
        vec![LiteralValue::Unknown; 3],
        1,
    );
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn typed_option_identity_diagnoses_conflicts_and_unsupported_modes() {
    let prefix = vec![numeric(&[4, 1]), numeric(&[4, 1]), numeric(&[1, 1])];
    let string = ValueFact::scalar(ValueKindFact::String);
    let mut args = prefix.clone();
    args.extend([
        string.clone(),
        numeric(&[2, 1]),
        string.clone(),
        numeric(&[2, 1]),
    ]);
    let inferred = call(
        args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("TVals".into()),
            LiteralValue::Unknown,
            LiteralValue::String("XVals".into()),
            LiteralValue::Unknown,
        ],
        2,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-PERFCURVE-THRESHOLD-CONFLICT"));
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));

    let mut args = prefix;
    args.extend([string, numeric(&[1, 1])]);
    let inferred = call(
        args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("NBoot".into()),
            LiteralValue::Number(2.0),
        ],
        1,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-PERFCURVE-UNSUPPORTED"));
}

#[test]
fn requested_outputs_respect_public_seven_output_limit() {
    let args = vec![numeric(&[4, 1]), numeric(&[4, 1]), numeric(&[1, 1])];
    let discarded = call(args.clone(), vec![LiteralValue::Unknown; 3], 0);
    assert!(discarded.outputs.is_empty());
    let excessive = call(args, vec![LiteralValue::Unknown; 3], 8);
    assert!(!excessive.diagnostics.is_empty());
}

#[test]
fn all_public_option_spellings_decode_once_into_typed_identities() {
    assert_eq!(
        PerfcurveOptionName::from_public("X_Crit"),
        Some(PerfcurveOptionName::XCrit)
    );
    assert_eq!(
        PerfcurveCriterion::from_public("tp+fp"),
        Some(PerfcurveCriterion::PredictedPositive)
    );
    assert_eq!(
        PerfcurveProcessNaN::from_public("addtofalse"),
        Some(PerfcurveProcessNaN::AddToFalse)
    );
    assert!(PerfcurveOptionName::from_public("x-crit").is_none());
    assert_eq!(
        PerfcurveBoolean::from_public("On"),
        Some(PerfcurveBoolean::True)
    );
}

#[test]
fn invalid_literal_logical_control_has_no_proven_output() {
    let request = call(
        vec![
            numeric(&[4, 1]),
            numeric(&[4, 1]),
            numeric(&[1, 1]),
            ValueFact::scalar(ValueKindFact::String),
            numeric(&[1, 1]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("UseNearest".into()),
            LiteralValue::Number(2.0),
        ],
        1,
    );
    assert!(request
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PERFCURVE-NEAREST"));
    assert!(matches!(request.outputs[0].kind, ValueKindFact::Unknown));
}
