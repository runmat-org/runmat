use super::*;
use runmat_types::{
    DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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

fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: args,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn canonical_entries_have_one_binding_and_documentation() {
    assert_eq!(ENTRIES.len(), 7);
    for entry in ENTRIES {
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
    assert!(!BUSDAYS_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::Clock));
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

    let inferencers: &[(
        &BuiltinCatalogEntry,
        fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    )] = &[
        (&DAYS252BUS_CATALOG_ENTRY, inference::days252bus),
        (&DAYSDIF_CATALOG_ENTRY, inference::daysdif),
        (&FBUSDATE_CATALOG_ENTRY, inference::fbusdate),
        (&LBUSDATE_CATALOG_ENTRY, inference::lbusdate),
    ];
    for (entry, infer) in inferencers {
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
