use super::*;
use crate::*;
use runmat_types::*;

fn number(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn call(args: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: args,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn every_legacy_identity_has_one_catalog_binding_and_executable_docs() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(entries.len(), 7);
    for entry in entries {
        assert_eq!(entry.bindings, &REQUIRED_DEFAULT_BINDING);
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
    }
    assert!(DATENUM_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::Clock));
    assert!(contracts::DATESTR.signatures.iter().any(|signature| {
        signature
            .inputs
            .get(1)
            .is_some_and(|format| format.ty == BuiltinParamType::NumericScalar)
    }));
    assert!(!EOMDAY_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::Clock));
    assert!(DATENUM_CATALOG_ENTRY
        .extensions
        .iter()
        .any(|extension| extension.id == DATENUM_PARTIAL_COMPONENT_EXTENSION.id));
    assert!(ISBETWEEN_CATALOG_ENTRY
        .extensions
        .iter()
        .any(|extension| extension.id == ISBETWEEN_SERIAL_DATE_EXTENSION.id));
}

#[test]
fn datenum_distinguishes_date_vector_rows_from_serial_array_shapes() {
    let two_rows = number(ShapeFact::from(vec![Some(2), Some(3)]));
    let result = inference::datenum(&call(vec![two_rows], 1), &DATENUM_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    let serials = number(ShapeFact::from(vec![Some(2), Some(2)]));
    let result = inference::datenum(&call(vec![serials], 1), &DATENUM_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    let unknown_columns = number(ShapeFact::from(vec![Some(2), None]));
    let datenum = inference::datenum(
        &call(vec![unknown_columns.clone()], 1),
        &DATENUM_CATALOG_ENTRY,
    );
    assert_eq!(datenum.outputs[0].shape, ShapeFact::Unknown);
    let datestr = inference::datestr(&call(vec![unknown_columns], 1), &DATESTR_CATALOG_ENTRY);
    assert_eq!(datestr.outputs[0].shape.known_dims().unwrap()[0], None);
}

#[test]
fn datevec_requested_outputs_switch_between_matrix_and_component_columns() {
    let serials = number(ShapeFact::from(vec![Some(2), Some(1)]));
    let one = inference::datevec(&call(vec![serials.clone()], 1), &DATEVEC_CATALOG_ENTRY);
    assert_eq!(
        one.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(6)])
    );
    let two = inference::datevec(&call(vec![serials.clone()], 2), &DATEVEC_CATALOG_ENTRY);
    assert_eq!(two.outputs.len(), 2);
    assert_eq!(
        two.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        two.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    let invalid = inference::datevec(&call(vec![serials], 7), &DATEVEC_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATEVEC-OUTPUTS"));
}

#[test]
fn weekday_names_are_character_rows_and_isbetween_is_logical() {
    let serials = number(ShapeFact::from(vec![Some(2), Some(1)]));
    let weekday = inference::weekday(&call(vec![serials.clone()], 2), &WEEKDAY_CATALOG_ENTRY);
    assert_eq!(weekday.outputs[0].shape, serials.shape);
    assert_eq!(weekday.outputs[1].kind, ValueKindFact::Character);
    assert_eq!(weekday.outputs[1].shape.known_dims().unwrap()[0], Some(2));
    let between = inference::isbetween(
        &call(
            vec![
                serials,
                number(ShapeFact::Scalar),
                number(ShapeFact::Scalar),
            ],
            1,
        ),
        &ISBETWEEN_CATALOG_ENTRY,
    );
    assert_eq!(between.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        between.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn invalid_component_and_date_input_are_diagnosed() {
    // Numeric datestr selectors are a typed overload, not arbitrary numeric text.
    for (code, accepted) in [
        (-1.0, true),
        (0.0, true),
        (31.0, true),
        (-2.0, false),
        (32.0, false),
        (0.5, false),
        (f64::NAN, false),
        (f64::INFINITY, false),
    ] {
        let mut request = call(
            vec![number(ShapeFact::Scalar), number(ShapeFact::Scalar)],
            1,
        );
        request.literals =
            LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(code)]);
        let inferred = inference::datestr(&request, &DATESTR_CATALOG_ENTRY);
        assert_eq!(inferred.diagnostics.is_empty(), accepted, "code {code}");
    }
    let mut device = number(ShapeFact::Scalar);
    device.residency = ResidencyFact::Device { provider: None };
    let mut remote = number(ShapeFact::Scalar);
    remote.residency = ResidencyFact::Remote { pool: None };
    for selector in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
        ValueFact::scalar(ValueKindFact::Logical),
        number(ShapeFact::from(vec![Some(1), Some(2)])),
        device,
        remote,
    ] {
        let inferred = inference::datestr(
            &call(vec![number(ShapeFact::Scalar), selector], 1),
            &DATESTR_CATALOG_ENTRY,
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DATESTR-FORMAT"));
    }
    for selector in [
        number(ShapeFact::Unknown),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ] {
        assert!(inference::datestr(
            &call(vec![number(ShapeFact::Scalar), selector], 1),
            &DATESTR_CATALOG_ENTRY
        )
        .diagnostics
        .is_empty());
    }
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let bad = inference::eomday(
        &call(vec![complex, number(ShapeFact::Scalar)], 1),
        &EOMDAY_CATALOG_ENTRY,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATETIME-LEGACY-NUMERIC"));
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = inference::datenum(&call(vec![unknown], 1), &DATENUM_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn weekday_rejects_invalid_known_day_form() {
    let mut request = call(
        vec![
            number(ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::String),
        ],
        2,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("other".into()),
    ]);
    let result = inference::weekday(&request, &WEEKDAY_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-WEEKDAY-FORM"));
}

#[test]
fn weekday_accepts_case_insensitive_day_form_literals() {
    let mut request = call(
        vec![
            number(ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::String),
        ],
        2,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("LONG".into()),
    ]);
    let result = inference::weekday(&request, &WEEKDAY_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn known_other_object_is_not_a_datetime_input() {
    let other = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: Some(ClassId(7)),
            runtime_class: None,
            properties: std::collections::BTreeMap::new(),
            properties_complete: false,
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let result = inference::datenum(&call(vec![other], 1), &DATENUM_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATETIME-LEGACY-INPUT"));
}

#[test]
fn date_arrays_expand_scalars_but_not_distinct_non_scalar_shapes() {
    let column = number(ShapeFact::from(vec![Some(2), Some(1)]));
    let row = number(ShapeFact::from(vec![Some(1), Some(3)]));
    let eomday = inference::eomday(
        &call(vec![column.clone(), row.clone()], 1),
        &EOMDAY_CATALOG_ENTRY,
    );
    assert!(eomday
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-EOMDAY-SHAPE"));
    let two_date_vectors = number(ShapeFact::from(vec![Some(2), Some(6)]));
    let three_date_vectors = number(ShapeFact::from(vec![Some(3), Some(6)]));
    let etime = inference::etime(
        &call(vec![two_date_vectors, three_date_vectors], 1),
        &ETIME_CATALOG_ENTRY,
    );
    assert!(etime
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-ETIME-SHAPE"));
    let between = inference::isbetween(
        &call(vec![column.clone(), number(ShapeFact::Scalar), row], 1),
        &ISBETWEEN_CATALOG_ENTRY,
    );
    assert!(between.diagnostics.is_empty());
    assert_eq!(
        between.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    let incompatible = inference::isbetween(
        &call(
            vec![
                column.clone(),
                number(ShapeFact::from(vec![Some(3), Some(1)])),
                number(ShapeFact::Scalar),
            ],
            1,
        ),
        &ISBETWEEN_CATALOG_ENTRY,
    );
    assert!(incompatible
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-ISBETWEEN-SHAPE"));
    let empty = inference::isbetween(
        &call(
            vec![
                number(ShapeFact::from(vec![Some(0), Some(1)])),
                number(ShapeFact::from(vec![Some(1), Some(3)])),
                number(ShapeFact::Scalar),
            ],
            1,
        ),
        &ISBETWEEN_CATALOG_ENTRY,
    );
    assert_eq!(
        empty.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(3)])
    );
    let scalar_bound = inference::isbetween(
        &call(
            vec![column.clone(), number(ShapeFact::Scalar), column.clone()],
            1,
        ),
        &ISBETWEEN_CATALOG_ENTRY,
    );
    assert!(scalar_bound.diagnostics.is_empty());
    assert_eq!(scalar_bound.outputs[0].shape, column.shape);
}
