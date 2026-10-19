use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn request(x: ValueFact, y: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![x, y],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn grouped_counts_prove_double_coefficient_geometry() {
    let result = inference::infer(
        &request(numeric(4, 2), numeric(4, 3), 3),
        &MNRFIT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(matches!(result.outputs[2].kind, ValueKindFact::Struct(_)));
}

fn with_constant(mut call: CallRequest, literal: LiteralValue, fact: ValueFact) -> CallRequest {
    call.arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    call.arguments.push(fact);
    call.literals
        .literal_args
        .extend([LiteralValue::String("Constant".into()), literal]);
    call
}

#[test]
fn grouped_constant_controls_coefficient_and_stats_geometry() {
    for (literal, fact, coefficients) in [
        (
            LiteralValue::String("off".into()),
            ValueFact::scalar(ValueKindFact::String),
            2,
        ),
        (
            LiteralValue::Bool(false),
            ValueFact::scalar(ValueKindFact::Logical),
            2,
        ),
        (LiteralValue::Number(1.0), numeric(1, 1), 3),
    ] {
        let call = with_constant(request(numeric(4, 2), numeric(4, 3), 3), literal, fact);
        let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        let b_shape = ShapeFact::from(vec![Some(coefficients), Some(2)]);
        assert_eq!(result.outputs[0].shape, b_shape);
        let ValueKindFact::Struct(stats) = &result.outputs[2].kind else {
            panic!("stats should be a typed struct");
        };
        for field in ["beta", "se", "t", "p"] {
            assert_eq!(stats.fields[field].shape, b_shape, "{field}");
        }
        for field in ["covb", "coeffcorr"] {
            assert_eq!(
                stats.fields[field].shape,
                ShapeFact::from(vec![Some(coefficients * 2), Some(coefficients * 2)]),
                "{field}"
            );
        }
    }
}

#[test]
fn unknown_or_invalid_constant_never_proves_grouped_geometry() {
    for (literal, fact, invalid) in [
        (
            LiteralValue::Unknown,
            ValueFact::scalar(ValueKindFact::String),
            false,
        ),
        (
            LiteralValue::String("bogus".into()),
            ValueFact::scalar(ValueKindFact::String),
            true,
        ),
        (LiteralValue::Number(2.0), numeric(1, 1), true),
    ] {
        let call = with_constant(request(numeric(4, 2), numeric(4, 3), 1), literal, fact);
        let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
        assert_eq!(result.diagnostics.is_empty(), !invalid);
        assert_eq!(result.outputs[0].kind == ValueKindFact::Unknown, invalid);
        if !invalid {
            assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
        }
    }
}

#[test]
fn vector_response_category_count_remains_dynamic() {
    let result = inference::infer(
        &request(numeric(4, 2), numeric(4, 1), 1),
        &MNRFIT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
}

#[test]
fn requested_outputs_beyond_stats_follow_runtime_zero_padding() {
    let result = inference::infer(
        &request(numeric(4, 2), numeric(4, 3), 4),
        &MNRFIT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 4);
    assert_eq!(
        result.outputs[3].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[3].shape, ShapeFact::Scalar);
}

#[test]
fn impossible_known_matrix_and_response_shapes_never_prove_success() {
    let mut empty_x = numeric(4, 2);
    empty_x.shape = ShapeFact::from(vec![Some(0), None]);
    let mut component_x = numeric(4, 2);
    component_x.shape = ShapeFact::from(vec![Some(4), Some(2), Some(2)]);
    let cases = [
        (empty_x, numeric(4, 1), "RM-CATALOG-MNRFIT-X-EMPTY"),
        (component_x, numeric(4, 1), "RM-CATALOG-MNRFIT-X-SHAPE"),
        (numeric(4, 1), numeric(2, 2), "RM-CATALOG-MNRFIT-Y-SHAPE"),
        (
            numeric(4, 1),
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::from(vec![Some(2), Some(2)]),
                StorageFact::Dense,
            ),
            "RM-CATALOG-MNRFIT-Y-SHAPE",
        ),
        (
            numeric(4, 1),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(3), Some(5)]),
                StorageFact::Dense,
            ),
            "RM-CATALOG-MNRFIT-Y-SHAPE",
        ),
    ];
    for (x, y, diagnostic) in cases {
        let result = inference::infer(&request(x, y, 1), &MNRFIT_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|item| item.code == diagnostic));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}

#[test]
fn known_sparse_predictors_and_responses_do_not_prove_a_fit() {
    for (argument, code) in [
        (0, "RM-CATALOG-MNRFIT-X-STORAGE"),
        (1, "RM-CATALOG-MNRFIT-Y-STORAGE"),
    ] {
        let mut call = request(numeric(4, 2), numeric(4, 3), 3);
        call.arguments[argument].storage = StorageFact::Sparse;
        let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
        assert!(result.diagnostics.iter().any(|item| item.code == code));
        assert!(result
            .outputs
            .iter()
            .all(|value| value.kind == ValueKindFact::Unknown));
    }
}

#[test]
fn trailing_singletons_and_unknown_extents_remain_admissible() {
    let mut x = numeric(4, 2);
    x.shape = ShapeFact::from(vec![Some(4), Some(2), Some(1)]);
    let result = inference::infer(&request(x, numeric(4, 3), 1), &MNRFIT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());

    let mut y = numeric(4, 3);
    y.shape = ShapeFact::from(vec![Some(4), Some(3), None]);
    let result = inference::infer(&request(numeric(4, 2), y, 1), &MNRFIT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn invalid_known_predictor_or_option_does_not_prove_success() {
    let mut call = request(ValueFact::scalar(ValueKindFact::String), numeric(4, 1), 1);
    call.arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    call.arguments.push(numeric(1, 1));
    call.literals.literal_args.extend([
        LiteralValue::String("unknown".into()),
        LiteralValue::Number(1.0),
    ]);
    let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-MNRFIT-X"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-MNRFIT-OPTION"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn public_options_decode_once_to_typed_identities() {
    assert_eq!(
        MnrfitOption::from_public("ITERATIONLIMIT"),
        Some(MnrfitOption::IterationLimit)
    );
    assert_eq!(MnrfitOption::from_public("bogus"), None);
}

#[test]
fn known_invalid_option_value_kinds_and_shapes_never_prove_a_fit() {
    let cases = [
        (
            "Weights",
            ValueFact::scalar(ValueKindFact::String),
            "RM-CATALOG-MNRFIT-OPTION-VALUE",
        ),
        ("Weights", numeric(2, 2), "RM-CATALOG-MNRFIT-OPTION-VALUE"),
        (
            "IterationLimit",
            numeric(1, 2),
            "RM-CATALOG-MNRFIT-OPTION-VALUE",
        ),
        ("Options", numeric(1, 1), "RM-CATALOG-MNRFIT-OPTION-VALUE"),
    ];
    for (name, value, expected) in cases {
        let mut call = request(numeric(4, 1), numeric(4, 1), 1);
        call.arguments
            .push(ValueFact::scalar(ValueKindFact::String));
        call.arguments.push(value);
        call.literals
            .literal_args
            .extend([LiteralValue::String(name.into()), LiteralValue::Unknown]);
        let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
        assert!(result.diagnostics.iter().any(|item| item.code == expected));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}

#[test]
fn scalar_character_controls_and_unknown_values_remain_admissible() {
    let mut call = request(numeric(4, 1), numeric(4, 1), 1);
    call.arguments.push(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(8)]),
        StorageFact::Dense,
    ));
    call.arguments.push(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    ));
    call.literals.literal_args.extend([
        LiteralValue::Character("Constant".into()),
        LiteralValue::Character("on".into()),
    ]);
    let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());

    call.arguments[3] = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    call.literals.literal_args[3] = LiteralValue::Unknown;
    let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn definite_nontext_and_nonscalar_option_names_are_diagnosed() {
    for invalid_name in [
        numeric(1, 1),
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ),
    ] {
        let mut call = request(numeric(4, 1), numeric(4, 1), 1);
        call.arguments.push(invalid_name);
        call.arguments.push(numeric(1, 1));
        call.literals
            .literal_args
            .extend([LiteralValue::Unknown, LiteralValue::Unknown]);
        let result = inference::infer(&call, &MNRFIT_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-CATALOG-MNRFIT-OPTION"));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}
