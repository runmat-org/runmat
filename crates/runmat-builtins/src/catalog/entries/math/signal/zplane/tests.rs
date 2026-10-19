use super::*;
use runmat_types::*;

fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn call(args: Vec<ValueFact>, outputs: usize) -> CallInference {
    let literals = vec![LiteralValue::Unknown; args.len()];
    inference::infer(
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &ZPLANE_CATALOG_ENTRY,
    )
}

#[test]
fn logical_sos_and_complex_integer_vectors_have_no_success_fact() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(6)]),
        StorageFact::Dense,
    );
    let sos = call(vec![logical], 1);
    assert!(sos
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ZPLANE-SOS-CLASS"));
    assert_eq!(sos.outputs[0].kind, ValueKindFact::Unknown);

    let complex_integer = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let vectors = call(
        vec![
            complex_integer,
            numeric(ShapeFact::from(vec![Some(1), Some(2)])),
        ],
        1,
    );
    assert!(vectors
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ZPLANE-COMPLEX-INTEGER"));
    assert_eq!(vectors.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn plotting_and_assigned_forms_share_host_handle_contract() {
    let sos = numeric(ShapeFact::from(vec![Some(2), Some(6)]));
    let plotted = call(vec![sos.clone()], 0);
    assert!(plotted.diagnostics.is_empty(), "{:?}", plotted.diagnostics);
    assert!(plotted.outputs.is_empty());
    let assigned = call(vec![sos], 1);
    assert!(
        assigned.diagnostics.is_empty(),
        "{:?}",
        assigned.diagnostics
    );
    assert_eq!(
        assigned.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(assigned.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        assigned.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn invalid_sos_and_gain_are_diagnosed_before_host_handle_fact() {
    let bad_sos = call(vec![numeric(ShapeFact::from(vec![Some(2), Some(5)]))], 1);
    assert!(!bad_sos.diagnostics.is_empty());
    assert_eq!(bad_sos.outputs[0].kind, ValueKindFact::Unknown);
    let bad_gain = call(
        vec![
            numeric(ShapeFact::from(vec![Some(2), Some(1)])),
            numeric(ShapeFact::from(vec![Some(2), Some(1)])),
            numeric(ShapeFact::from(vec![Some(1), Some(2)])),
        ],
        1,
    );
    assert!(!bad_gain.diagnostics.is_empty());
    assert_eq!(bad_gain.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn unknown_filter_data_and_requested_padding_do_not_invent_roots() {
    let unknown = call(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1);
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(
        unknown.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    let padded = call(vec![numeric(ShapeFact::from(vec![Some(1), Some(6)]))], 2);
    assert!(padded.diagnostics.is_empty());
    assert_eq!(padded.outputs.len(), 2);
    assert_eq!(
        padded.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn leading_axes_three_argument_form_is_not_misread_as_zpk_gain() {
    let axes = numeric(ShapeFact::Scalar);
    let numerator = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let denominator = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let target = call(vec![axes, numerator, denominator], 1);
    assert!(target.diagnostics.is_empty(), "{:?}", target.diagnostics);
    assert_eq!(target.outputs[0].residency, ResidencyFact::Host);

    let explicit_zpk = call(
        vec![
            numeric(ShapeFact::from(vec![Some(3), Some(1)])),
            numeric(ShapeFact::from(vec![Some(2), Some(1)])),
            numeric(ShapeFact::from(vec![Some(1), Some(2)])),
        ],
        1,
    );
    assert!(!explicit_zpk.diagnostics.is_empty());
}

#[test]
fn known_inexact_integer_filter_literal_has_no_handle_success_fact() {
    let result = inference::infer(
        &CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::UInt64,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::Scalar,
                    StorageFact::Scalar,
                ),
                numeric(ShapeFact::from(vec![Some(1), Some(2)])),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Integer {
                    text: "9007199254740993".into(),
                    class: NumericClass::UInt64,
                },
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &ZPLANE_CATALOG_ENTRY,
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
