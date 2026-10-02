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
fn ambiguous_scalar_axes_cannot_prove_missing_y_or_mismatched_xy() {
    let result = call(
        vec![real(vec![Some(1), Some(1)]), real(vec![Some(2), Some(1)])],
        vec![],
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    let three = call(
        vec![
            real(vec![Some(1), Some(1)]),
            real(vec![Some(2), Some(1)]),
            real(vec![Some(1), Some(1)]),
        ],
        vec![],
    );
    assert!(!three.diagnostics.is_empty());
    assert!(matches!(three.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn nonscalar_string_option_value_is_known_invalid() {
    let option = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let string_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let result = call(
        vec![
            real(vec![Some(3), Some(1)]),
            real(vec![Some(3), Some(1)]),
            option,
            string_array,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("DisplayName".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
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

#[test]
fn table_selectors_and_option_forms_have_truthful_structured_signatures() {
    let signatures = BINSCATTER_CATALOG_ENTRY.descriptor.signatures;
    let table = signatures
        .iter()
        .find(|signature| signature.label == "binscatter(tbl, xvar, yvar)")
        .expect("table signature");
    assert_eq!(table.inputs[1].ty, BuiltinParamType::Any);
    assert_eq!(table.inputs[2].ty, BuiltinParamType::Any);
    for label in [
        "binscatter(tbl, xvar, yvar, Name, Value)",
        "h = binscatter(x, y, Name, Value)",
        "h = binscatter(tbl, xvar, yvar, nbins)",
        "h = binscatter(ax, x, y, Name, Value)",
    ] {
        assert!(signatures.iter().any(|signature| signature.label == label));
    }
}

#[test]
fn retained_legacy_examples_exercise_distinct_bin_and_table_forms() {
    let examples = BINSCATTER_CATALOG_ENTRY.documentation.examples;
    let limits = examples
        .iter()
        .find(|example| example.id == "binscatter-limits")
        .expect("limits example");
    assert!(limits.program.contains("'NumBins',[20 10]"));
    assert!(limits.program.contains("'YLimits',[-1 1]"));
    let table = examples
        .iter()
        .find(|example| example.id == "binscatter-table")
        .expect("table example");
    assert!(table.program.contains("binscatter(T,'x','y',25)"));
    assert!(matches!(
        limits.verification,
        BuiltinExampleVerification::Figure { .. }
    ));
    assert!(matches!(
        table.verification,
        BuiltinExampleVerification::Figure { .. }
    ));
}
