use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn real(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(literals),
            arguments: args,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &BINSCATTER_CATALOG_ENTRY,
    )
}

#[test]
fn valid_xy_proves_scalar_double_chart_handle() {
    let result = call(
        vec![real(vec![Some(4), Some(1)]), real(vec![Some(4), Some(1)])],
        vec![],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert!(result
        .effects
        .0
        .contains(&runmat_types::EffectKind::UserInterface));
}

#[test]
fn wrong_vector_geometry_and_lengths_do_not_prove_output() {
    let bad_shape = call(
        vec![real(vec![Some(2), Some(2)]), real(vec![Some(4), Some(1)])],
        vec![],
    );
    assert!(!bad_shape.diagnostics.is_empty());
    assert!(matches!(bad_shape.outputs[0].kind, ValueKindFact::Unknown));
    let bad_lengths = call(
        vec![real(vec![Some(4), Some(1)]), real(vec![Some(3), Some(1)])],
        vec![],
    );
    assert!(!bad_lengths.diagnostics.is_empty());
}

#[test]
fn known_unsupported_option_name_uses_typed_grammar() {
    let option = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(
        vec![
            real(vec![Some(4), Some(1)]),
            real(vec![Some(4), Some(1)]),
            option,
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Normalization".into()),
            LiteralValue::Number(1.0),
        ],
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn known_bad_bin_count_and_host_option_values_do_not_prove_a_handle() {
    let option = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let bad_positional = call(
        vec![
            real(vec![Some(4), Some(1)]),
            real(vec![Some(4), Some(1)]),
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
        ],
    );
    assert!(!bad_positional.diagnostics.is_empty());
    assert!(matches!(
        bad_positional.outputs[0].kind,
        ValueKindFact::Unknown
    ));

    let bad_alpha = call(
        vec![
            real(vec![Some(4), Some(1)]),
            real(vec![Some(4), Some(1)]),
            option,
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("FaceAlpha".into()),
            LiteralValue::Number(2.0),
        ],
    );
    assert!(!bad_alpha.diagnostics.is_empty());
    assert!(matches!(bad_alpha.outputs[0].kind, ValueKindFact::Unknown));
}
