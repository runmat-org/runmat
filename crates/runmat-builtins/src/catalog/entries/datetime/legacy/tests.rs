use super::*;
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
    assert_eq!(ENTRIES.len(), 7);
    for entry in ENTRIES {
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
