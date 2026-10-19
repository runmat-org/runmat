use super::*;
use runmat_types::{
    CallRequest, DimensionFact, EffectKind, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

fn sample(shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn infer_ecdf(
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> runmat_types::CallInference {
    inference::ecdf(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &ECDF_CATALOG_ENTRY,
    )
}

#[test]
fn ecdf_numeric_outputs_are_double_columns_without_graphics_effects() {
    let result = infer_ecdf(vec![sample(&[4, 1])], vec![LiteralValue::Unknown], 4);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 4);
    for output in &result.outputs {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
        assert_eq!(
            output.shape,
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)]
            }
        );
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    assert!(!result.effects.0.contains(&EffectKind::UserInterface));
}

#[test]
fn ecdf_no_output_form_retains_graphics_effect_and_valid_typed_options() {
    let name = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::String);
    let result = infer_ecdf(
        vec![sample(&[4, 1]), name, value],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Function".into()),
            LiteralValue::String("survivor".into()),
        ],
        0,
    );
    assert!(result.diagnostics.is_empty());
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
    assert!(result.outputs.is_empty());
}

#[test]
fn known_invalid_ecdf_inputs_and_controls_never_prove_results() {
    let result = infer_ecdf(vec![sample(&[2, 2])], vec![], 1);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));

    let result = infer_ecdf(
        vec![
            sample(&[4, 1]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Bounds".into()),
            LiteralValue::String("invalid".into()),
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));

    let result = infer_ecdf(
        vec![
            sample(&[4, 1]),
            ValueFact::scalar(ValueKindFact::String),
            sample(&[2, 1]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Frequency".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FREQUENCY"));
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn cdfplot_proves_host_handle_and_five_stat_fields() {
    let result = inference::cdfplot(
        &CallRequest {
            arguments: vec![sample(&[4, 1])],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
        &CDFPLOT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Struct(fields) = &result.outputs[1].kind else {
        panic!("expected statistics struct");
    };
    assert!(fields.fields_complete);
    for name in ["min", "max", "mean", "median", "std"] {
        assert_eq!(
            fields.fields[name].numeric().unwrap().class,
            NumericClass::Double
        );
    }
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
}

#[test]
fn cdfplot_known_matrix_is_diagnosed_without_claiming_a_handle() {
    for shape in [&[2, 2][..], &[1, 1, 4][..]] {
        let result = inference::cdfplot(
            &CallRequest {
                arguments: vec![sample(shape)],
                literals: LiteralContext::new(vec![LiteralValue::Unknown]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
            &CDFPLOT_CATALOG_ENTRY,
        );
        assert!(!result.diagnostics.is_empty(), "shape {shape:?}");
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
        assert!(matches!(result.outputs[1].kind, ValueKindFact::Unknown));
    }
}

#[test]
fn ambiguous_axes_forms_and_known_invalid_controls_never_prove_outputs() {
    let vector = sample(&[2, 1]);
    let numeric_scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let option_name = ValueFact::scalar(ValueKindFact::String);
    let cases = [
        infer_ecdf(
            vec![vector.clone(), vector.clone()],
            vec![LiteralValue::Unknown; 2],
            1,
        ),
        infer_ecdf(
            vec![numeric_scalar.clone(), vector.clone()],
            vec![LiteralValue::Unknown; 2],
            1,
        ),
        infer_ecdf(
            vec![
                vector.clone(),
                option_name.clone(),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("Alpha".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
        infer_ecdf(
            vec![
                vector.clone(),
                option_name.clone(),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Complex,
                })),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("Alpha".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
        infer_ecdf(
            vec![vector.clone(), option_name.clone(), sample(&[2, 2])],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("Frequency".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
        infer_ecdf(
            vec![
                vector.clone(),
                option_name,
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Complex,
                    }),
                    ShapeFact::from(vec![Some(2), Some(1)]),
                    StorageFact::Dense,
                ),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("Censoring".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
    ];
    for result in cases {
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    }
    let cdfplot = |first: ValueFact| {
        inference::cdfplot(
            &CallRequest {
                arguments: vec![first, vector.clone()],
                literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
            &CDFPLOT_CATALOG_ENTRY,
        )
    };
    for result in [cdfplot(vector.clone()), cdfplot(numeric_scalar)] {
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
        assert!(matches!(result.outputs[1].kind, ValueKindFact::Unknown));
    }
}
