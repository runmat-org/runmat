use super::*;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric(class: NumericClass, rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn request(
    sample: ValueFact,
    training: ValueFact,
    group: ValueFact,
    outputs: usize,
) -> CallRequest {
    CallRequest {
        arguments: vec![sample, training, group],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 3]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn exact_integer_labels_and_numerical_outputs_have_distinct_facts() {
    let result = inference::infer(
        &request(
            numeric(NumericClass::Single, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            numeric(NumericClass::UInt64, 4, 1),
            5,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    for output in &result.outputs[1..4] {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
    }
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(2), None])
    );
    assert_eq!(result.outputs[4].shape, ShapeFact::from(vec![None, None]));
    let ValueKindFact::Struct(coeff) = &result.outputs[4].kind else {
        panic!("coefficient output must be a structure array");
    };
    assert!(coeff.fields_complete);
    assert!(coeff.fields.contains_key("const"));
    assert!(coeff.fields.contains_key("linear"));
    assert!(!coeff.fields.contains_key("quadratic"));
    for name in ["name1", "name2"] {
        assert_eq!(
            coeff.fields[name].numeric().unwrap().class,
            NumericClass::UInt64
        );
        assert_eq!(
            coeff.fields[name].shape,
            ShapeFact::from(vec![Some(1), Some(1)])
        );
    }
}

#[test]
fn coefficient_structure_name_fields_retain_string_label_class() {
    let result = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(4), Some(1)]),
                StorageFact::Dense,
            ),
            5,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Struct(coeff) = &result.outputs[4].kind else {
        panic!("coefficient output must be a structure array");
    };
    for name in ["name1", "name2"] {
        assert_eq!(coeff.fields[name].kind, ValueKindFact::String);
        assert_eq!(
            coeff.fields[name].shape,
            ShapeFact::from(vec![Some(1), Some(1)])
        );
    }
}

#[test]
fn coefficient_structure_fact_tracks_conditional_quadratic_field() {
    let mut quadratic = request(
        numeric(NumericClass::Double, 2, 1),
        numeric(NumericClass::Double, 6, 1),
        numeric(NumericClass::Double, 6, 1),
        5,
    );
    quadratic
        .arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    quadratic
        .literals
        .literal_args
        .push(LiteralValue::String("quadratic".into()));
    let result = inference::infer(&quadratic, &CLASSIFY_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Struct(coeff) = &result.outputs[4].kind else {
        panic!("quadratic coefficients must be structure arrays");
    };
    assert!(coeff.fields_complete);
    assert!(coeff.fields.contains_key("quadratic"));
    assert_eq!(
        coeff.fields["quadratic"].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );

    quadratic.literals.literal_args[3] = LiteralValue::Unknown;
    let dynamic = inference::infer(&quadratic, &CLASSIFY_CATALOG_ENTRY);
    let ValueKindFact::Struct(coeff) = &dynamic.outputs[4].kind else {
        panic!("dynamic type retains structure class");
    };
    assert!(!coeff.fields_complete);
    assert!(!coeff.fields.contains_key("quadratic"));
}

#[test]
fn invalid_geometry_does_not_prove_successful_outputs() {
    let result = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 2),
            numeric(NumericClass::Double, 4, 1),
            numeric(NumericClass::Double, 4, 1),
            1,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.CLASSIFY.COLUMNS"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn character_group_rows_count_labels_not_characters() {
    let result = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(4), Some(3)]),
                StorageFact::Dense,
            ),
            1,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), None])
    );
}

#[test]
fn label_array_families_require_vectors_but_admit_row_and_column_forms() {
    let families = [
        ValueKindFact::Logical,
        ValueKindFact::String,
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::String)),
            elements: Vec::new(),
            elements_complete: false,
        }),
    ];
    for family in families {
        for shape in [[1, 4], [4, 1]] {
            let result = inference::infer(
                &request(
                    numeric(NumericClass::Double, 2, 1),
                    numeric(NumericClass::Double, 4, 1),
                    ValueFact::proven(
                        family.clone(),
                        ShapeFact::from(shape.map(Some).to_vec()),
                        StorageFact::Dense,
                    ),
                    1,
                ),
                &CLASSIFY_CATALOG_ENTRY,
            );
            assert!(result.diagnostics.is_empty(), "{family:?} {shape:?}");
        }
        let result = inference::infer(
            &request(
                numeric(NumericClass::Double, 2, 1),
                numeric(NumericClass::Double, 4, 1),
                ValueFact::proven(
                    family.clone(),
                    ShapeFact::from(vec![Some(2), Some(2)]),
                    StorageFact::Dense,
                ),
                1,
            ),
            &CLASSIFY_CATALOG_ENTRY,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|item| item.code == "RM.CLASSIFY.GROUP"));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}

#[test]
fn nonscalar_string_controls_do_not_prove_outputs() {
    let base = request(
        numeric(NumericClass::Double, 2, 1),
        numeric(NumericClass::Double, 4, 1),
        numeric(NumericClass::Double, 4, 1),
        1,
    );
    for control_index in [3, 4] {
        for shape in [
            vec![Some(1), Some(2)],
            vec![Some(2), None],
            vec![Some(0), None],
        ] {
            let mut call = base.clone();
            call.arguments
                .resize_with(5, || ValueFact::scalar(ValueKindFact::String));
            call.literals.literal_args.resize(5, LiteralValue::Unknown);
            call.arguments[control_index] = ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(shape.clone()),
                StorageFact::Dense,
            );
            let result = inference::infer(&call, &CLASSIFY_CATALOG_ENTRY);
            let expected = if control_index == 3 {
                "RM.CLASSIFY.TYPE"
            } else {
                "RM.CLASSIFY.PRIOR"
            };
            assert!(
                result.diagnostics.iter().any(|item| item.code == expected),
                "{shape:?}"
            );
            assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        }
    }
}

#[test]
fn unknown_labels_and_excess_outputs_remain_conservative() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            unknown,
            6,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn discarded_outputs_still_admit_predictors_and_controls() {
    let valid = inference::infer(
        &request(
            numeric(NumericClass::Double, 2, 1),
            numeric(NumericClass::Double, 4, 1),
            numeric(NumericClass::Double, 4, 1),
            0,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(valid.outputs.is_empty());
    assert!(valid.diagnostics.is_empty());

    let invalid = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::Symbolic),
            numeric(NumericClass::Double, 4, 1),
            numeric(NumericClass::Double, 4, 1),
            0,
        ),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(invalid.outputs.is_empty());
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.CLASSIFY.INPUT"));

    let mut malformed_control = request(
        numeric(NumericClass::Double, 2, 1),
        numeric(NumericClass::Double, 4, 1),
        numeric(NumericClass::Double, 4, 1),
        0,
    );
    malformed_control
        .arguments
        .push(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })));
    malformed_control
        .literals
        .literal_args
        .push(LiteralValue::Number(42.0));
    let invalid = inference::infer(&malformed_control, &CLASSIFY_CATALOG_ENTRY);
    assert!(invalid.outputs.is_empty());
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.CLASSIFY.TYPE"));
}

#[test]
fn public_discriminant_spellings_decode_to_one_typed_identity() {
    assert_eq!(
        DiscriminantType::from_public("DIAGLINEAR"),
        Some(DiscriminantType::DiagLinear)
    );
    assert_eq!(
        DiscriminantType::from_public("mahalanobis"),
        Some(DiscriminantType::Mahalanobis)
    );
    assert_eq!(DiscriminantType::from_public("bogus"), None);
}

#[test]
fn invalid_known_prior_and_type_controls_do_not_prove_outputs() {
    let base = request(
        numeric(NumericClass::Double, 2, 1),
        numeric(NumericClass::Double, 4, 1),
        numeric(NumericClass::Double, 4, 1),
        1,
    );
    let text = |literal: LiteralValue| {
        let mut call = base.clone();
        call.arguments
            .push(ValueFact::scalar(ValueKindFact::String));
        call.arguments
            .push(ValueFact::scalar(ValueKindFact::String));
        call.literals
            .literal_args
            .extend([LiteralValue::String("linear".into()), literal]);
        call
    };
    let mut invalid = vec![text(LiteralValue::String("bogus".into()))];
    for prior in [
        ValueFact::scalar(ValueKindFact::Logical),
        numeric(NumericClass::Double, 1, 1),
        numeric(NumericClass::Double, 2, 2),
        ValueFact::proven(
            ValueKindFact::Struct(StructFact::scalar(BTreeMap::new(), true)),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        ),
    ] {
        let mut call = text(LiteralValue::Unknown);
        call.arguments[4] = prior;
        invalid.push(call);
    }
    for call in invalid {
        let result = inference::infer(&call, &CLASSIFY_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM.CLASSIFY.PRIOR"));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }

    let mut unknown_literal_type = text(LiteralValue::String("empirical".into()));
    unknown_literal_type.arguments[3] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    unknown_literal_type.literals.literal_args[3] = LiteralValue::Unknown;
    let result = inference::infer(&unknown_literal_type, &CLASSIFY_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.CLASSIFY.TYPE"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);

    let valid = inference::infer(
        &text(LiteralValue::String("empirical".into())),
        &CLASSIFY_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty());
    assert!(matches!(valid.outputs[0].kind, ValueKindFact::Numeric(_)));
}
