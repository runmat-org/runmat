use super::*;
use crate::*;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, EffectKind, LiteralContext, LiteralValue,
    NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

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

struct InferenceCase {
    entry: &'static BuiltinCatalogEntry,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
}

fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: args,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn canonical_entries_have_one_binding_and_documentation() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(entries.len(), 7);
    for entry in &entries {
        assert_eq!(entry.bindings, &REQUIRED_DEFAULT_BINDING);
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
        assert_eq!(
            entry.placement.portability,
            BuiltinPortability::NativeAndWasm
        );
    }
    assert!(HOLIDAYS_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::Clock));
    for entry in &entries {
        assert!(
            entry.contract.effects.contains(&EffectKind::Clock),
            "relative date/holiday text must retain Clock: {}",
            entry.identity.name
        );
    }
    for entry in &entries {
        for extension in [
            BUSINESS_DAY_TYPED_LEGACY_SERIAL_EXTENSION,
            BUSINESS_DAY_RESIDENT_INPUT_EXTENSION,
            BUSINESS_DAY_LOGICAL_NUMERIC_INPUT_EXTENSION,
        ] {
            assert!(
                entry
                    .extensions
                    .iter()
                    .any(|actual| actual.id == extension.id),
                "{} lacks {}",
                entry.identity.name,
                extension.id
            );
        }
    }
    assert!(HOLIDAYS_CATALOG_ENTRY
        .extensions
        .iter()
        .any(|extension| extension.id == HOLIDAYS_LOCAL_YEAR_EXTENSION.id));
}

#[test]
fn business_date_facts_preserve_known_shapes_and_host_residency() {
    let three = numeric(ShapeFact::from(vec![Some(1), Some(3)]));
    let scalar = numeric(ShapeFact::Scalar);
    let mask = inference::isbusday(
        &call(vec![three.clone()], vec![LiteralValue::Unknown]),
        &ISBUSDAY_CATALOG_ENTRY,
    );
    assert_eq!(mask.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(mask.outputs[0].shape, three.shape);

    let inferencers = &[
        InferenceCase {
            entry: &DAYS252BUS_CATALOG_ENTRY,
            infer: inference::days252bus,
        },
        InferenceCase {
            entry: &DAYSDIF_CATALOG_ENTRY,
            infer: inference::daysdif,
        },
        InferenceCase {
            entry: &FBUSDATE_CATALOG_ENTRY,
            infer: inference::fbusdate,
        },
        InferenceCase {
            entry: &LBUSDATE_CATALOG_ENTRY,
            infer: inference::lbusdate,
        },
    ];
    for InferenceCase { entry, infer } in inferencers {
        let result = infer(
            &call(
                vec![scalar.clone(), three.clone()],
                vec![LiteralValue::Unknown; 2],
            ),
            entry,
        );
        assert_eq!(result.outputs[0].shape, three.shape);
        assert!(matches!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                ..
            })
        ));
        assert_eq!(
            result.outputs[0].residency,
            runmat_types::ResidencyFact::Host
        );
    }
}

#[test]
fn invalid_options_are_diagnosed_without_inventing_dynamic_dates() {
    let date = numeric(ShapeFact::Scalar);
    let bad_basis = inference::daysdif(
        &call(
            vec![date.clone(), date.clone(), date.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Number(13.0),
            ],
        ),
        &DAYSDIF_CATALOG_ENTRY,
    );
    assert!(bad_basis
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DAYSDIF-BASIS"));

    let bad_mode = inference::busdays(
        &call(
            vec![date.clone(), date.clone(), date.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Number(0.0),
            ],
        ),
        &BUSDAYS_CATALOG_ENTRY,
    );
    assert!(bad_mode
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-BUSDAYS-PERIOD"));
    assert_eq!(bad_mode.outputs[0].shape, ShapeFact::Unknown);

    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = inference::busdays(
        &call(vec![unknown, date], vec![LiteralValue::Unknown; 2]),
        &BUSDAYS_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn busdays_does_not_prove_double_when_holiday_can_select_datetime_output() {
    let date = numeric(ShapeFact::Scalar);
    let mode = ValueFact::scalar(ValueKindFact::String);
    let datetime_holiday = ValueFact::scalar(ValueKindFact::Object(runmat_types::ObjectFact {
        class: None,
        runtime_class: Some(runmat_types::standard::DATETIME.owned()),
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: None,
    }));
    for holiday in [
        datetime_holiday,
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ] {
        let result = inference::busdays(
            &call(
                vec![date.clone(), date.clone(), mode.clone(), holiday],
                vec![LiteralValue::Unknown; 4],
            ),
            &BUSDAYS_CATALOG_ENTRY,
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }

    let serial_holiday = numeric(ShapeFact::Scalar);
    let result = inference::busdays(
        &call(
            vec![date.clone(), date, mode, serial_holiday],
            vec![LiteralValue::Unknown; 4],
        ),
        &BUSDAYS_CATALOG_ENTRY,
    );
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Numeric(_)));
}
