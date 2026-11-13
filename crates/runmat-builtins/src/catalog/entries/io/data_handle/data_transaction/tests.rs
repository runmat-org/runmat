use super::*;
use crate::catalog::inference::infer_catalog_call;
use runmat_types::*;
use std::collections::BTreeMap;

fn receiver() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::super::data_family::DATA_TRANSACTION_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn call(entry: &crate::BuiltinCatalogEntry, arguments: Vec<ValueFact>) -> CallInference {
    infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

#[test]
fn all_transaction_identities_and_commit_alias_are_unique_and_sorted() {
    let names: Vec<_> = ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(
        names,
        [
            "DataTransaction.abort",
            "DataTransaction.commit",
            "DataTransaction.create_array",
            "DataTransaction.delete_array",
            "DataTransaction.fill",
            "DataTransaction.id",
            "DataTransaction.resize",
            "DataTransaction.set_attr",
            "DataTransaction.set_attrs",
            "DataTransaction.status",
            "DataTransaction.write",
            "commit",
        ]
    );
    assert!(ENTRIES.iter().all(|entry| !entry.bindings.is_empty()));
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn queue_forms_admit_one_real_optional_slice_but_no_ignored_controls() {
    assert_eq!(DATATX_WRITE_DESCRIPTOR.signatures[0].inputs.len(), 4);
    assert_eq!(DATATX_RESIZE_DESCRIPTOR.signatures[0].inputs.len(), 3);
    assert_eq!(DATATX_FILL_DESCRIPTOR.signatures[0].inputs.len(), 4);
    let value = ValueFact::unknown(DynamicReason::RuntimeValue);
    for (entry, valid, invalid) in [
        (&DATATX_WRITE_CATALOG_ENTRY, 4, 5),
        (&DATATX_RESIZE_CATALOG_ENTRY, 3, 4),
        (&DATATX_FILL_CATALOG_ENTRY, 4, 5),
    ] {
        let args = [
            receiver(),
            value.clone(),
            value.clone(),
            value.clone(),
            value.clone(),
        ];
        assert!(call(entry, args[..valid].to_vec()).diagnostics.is_empty());
        assert!(call(entry, args[..invalid].to_vec())
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-DATA-TRANSACTION-ARITY"));
    }
}

#[test]
fn queue_and_commit_have_distinct_effects_and_exact_output_facts() {
    let key = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let queued = call(&DATATX_SET_ATTR_CATALOG_ENTRY, vec![receiver(), key, value]);
    assert!(queued.diagnostics.is_empty(), "{:?}", queued.diagnostics);
    assert!(matches!(queued.outputs[0].kind, ValueKindFact::Logical));
    assert!(queued.effects.0.contains(&EffectKind::EnvironmentWrite));
    assert!(!queued.effects.0.contains(&EffectKind::FilesystemWrite));
    assert_eq!(DATATX_SET_ATTR_CATALOG_ENTRY.integer_capabilities.len(), 1);

    let committed = call(&DATATX_COMMIT_CATALOG_ENTRY, vec![receiver()]);
    assert!(committed.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(committed.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(committed.effects.0.contains(&EffectKind::MaySuspend));
    assert_eq!(
        COMMIT_ALIAS_CATALOG_ENTRY.contract.effects,
        DATATX_COMMIT_CATALOG_ENTRY.contract.effects
    );
    let status = call(&DATATX_STATUS_CATALOG_ENTRY, vec![receiver()]);
    assert!(matches!(status.outputs[0].kind, ValueKindFact::String));
}

#[test]
fn known_invalid_receiver_controls_and_commit_options_diagnose_before_effects() {
    let forged = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::super::data_family::DATA_TRANSACTION_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    assert!(call(&DATATX_STATUS_CATALOG_ENTRY, vec![forged])
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-TRANSACTION-RECEIVER"));
    let bad_name = call(
        &DATATX_DELETE_ARRAY_CATALOG_ENTRY,
        vec![receiver(), ValueFact::scalar(ValueKindFact::Logical)],
    );
    assert!(bad_name
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-TRANSACTION-TEXT"));
    let bad_options = call(
        &DATATX_COMMIT_CATALOG_ENTRY,
        vec![receiver(), ValueFact::scalar(ValueKindFact::Logical)],
    );
    assert!(bad_options
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-TRANSACTION-STRUCT"));
    let bad_shape = call(
        &DATATX_RESIZE_CATALOG_ENTRY,
        vec![
            receiver(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
    );
    assert!(bad_shape
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-TRANSACTION-SHAPE"));
}
