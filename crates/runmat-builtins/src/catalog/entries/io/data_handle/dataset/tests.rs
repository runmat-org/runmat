use super::*;
use crate::catalog::inference::infer_catalog_call;
use runmat_types::*;
use std::collections::BTreeMap;

fn call(
    entry: &crate::BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    outputs: usize,
) -> CallInference {
    infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn dataset_identities_are_unique_sorted_and_source_linked() {
    let names: Vec<_> = ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(
        names,
        [
            "Dataset.array",
            "Dataset.arrays",
            "Dataset.attrs",
            "Dataset.begin",
            "Dataset.get_attr",
            "Dataset.has_array",
            "Dataset.id",
            "Dataset.path",
            "Dataset.refresh",
            "Dataset.set_attr",
            "Dataset.set_attrs",
            "Dataset.snapshot",
            "Dataset.version"
        ]
    );
    assert!(ENTRIES.iter().all(|entry| !entry.bindings.is_empty()));
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn optional_default_and_nonexistent_trailing_options_have_exact_arity() {
    assert_eq!(DATASET_GET_ATTR_DESCRIPTOR.signatures[0].inputs.len(), 3);
    assert_eq!(DATASET_BEGIN_DESCRIPTOR.signatures[0].inputs.len(), 1);
    assert_eq!(DATASET_SNAPSHOT_DESCRIPTOR.signatures[0].inputs.len(), 2);
    let receiver = super::super::inference::dataset();
    let control = ValueFact::scalar(ValueKindFact::String);
    for (entry, valid, invalid) in [
        (&DATASET_GET_ATTR_CATALOG_ENTRY, 3, 4),
        (&DATASET_SNAPSHOT_CATALOG_ENTRY, 2, 3),
        (&DATASET_BEGIN_CATALOG_ENTRY, 1, 2),
    ] {
        let args = [
            receiver.clone(),
            control.clone(),
            control.clone(),
            control.clone(),
        ];
        assert!(call(entry, args[..valid].to_vec(), 1)
            .diagnostics
            .is_empty());
        assert!(call(entry, args[..invalid].to_vec(), 1)
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-DATA-HANDLE-ARITY"));
    }
}

#[test]
fn attribute_and_lifecycle_forms_preserve_integer_and_effect_boundaries() {
    let receiver = super::super::inference::dataset();
    let text = ValueFact::scalar(ValueKindFact::String);
    let read = call(
        &DATASET_GET_ATTR_CATALOG_ENTRY,
        vec![receiver.clone(), text.clone()],
        1,
    );
    assert!(matches!(read.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(DATASET_GET_ATTR_CATALOG_ENTRY.integer_capabilities.len(), 2);
    let write = call(
        &DATASET_SET_ATTR_CATALOG_ENTRY,
        vec![
            receiver.clone(),
            text.clone(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
        ],
        1,
    );
    assert!(write.effects.0.contains(&EffectKind::Clock));
    assert!(write.effects.0.contains(&EffectKind::FilesystemWrite));
    assert_eq!(DATASET_SET_ATTR_CATALOG_ENTRY.integer_capabilities.len(), 1);
    let begin = call(&DATASET_BEGIN_CATALOG_ENTRY, vec![receiver.clone()], 1);
    let ValueKindFact::Object(tx) = &begin.outputs[0].kind else {
        panic!("transaction handle expected")
    };
    assert!(tx
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(super::super::super::data_family::DATA_TRANSACTION_CLASS)));
    assert_eq!(tx.handle_semantics, Some(true));
    assert!(begin.effects.0.contains(&EffectKind::EnvironmentWrite));
    let bad_attrs = call(&DATASET_SET_ATTRS_CATALOG_ENTRY, vec![receiver, text], 1);
    assert!(bad_attrs
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATA-HANDLE-ATTRS"));
}

#[test]
fn dataset_receiver_and_child_class_facts_are_sealed_handle_facts() {
    let receiver = super::super::inference::dataset();
    let child = call(
        &DATASET_ARRAY_CATALOG_ENTRY,
        vec![receiver.clone(), ValueFact::scalar(ValueKindFact::String)],
        1,
    );
    assert!(child.diagnostics.is_empty(), "{:?}", child.diagnostics);
    let ValueKindFact::Object(object) = &child.outputs[0].kind else {
        panic!("DataArray handle expected")
    };
    assert!(object
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(super::super::super::data_family::DATA_ARRAY_CLASS)));
    assert_eq!(object.handle_semantics, Some(true));
    assert_eq!(child.outputs[0].alias, AliasFact::Identity);
    assert_eq!(child.outputs[0].mutation, MutationFact::HandleSemantics);
    assert!(child.effects.0.contains(&EffectKind::FilesystemRead));

    let forged = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::super::data_family::DATASET_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let rejected = call(&DATASET_ARRAYS_CATALOG_ENTRY, vec![forged], 1);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATA-HANDLE-RECEIVER"));
    let unknown = call(
        &DATASET_ARRAYS_CATALOG_ENTRY,
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        1,
    );
    assert!(!unknown
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATA-HANDLE-RECEIVER"));
}

#[test]
fn known_invalid_text_arity_and_output_count_are_diagnosed() {
    let receiver = super::super::inference::dataset();
    let wrong_text = call(
        &DATASET_HAS_ARRAY_CATALOG_ENTRY,
        vec![receiver.clone(), ValueFact::scalar(ValueKindFact::Logical)],
        1,
    );
    assert!(wrong_text
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATA-HANDLE-TEXT"));
    let wrong_arity = call(
        &DATASET_ID_CATALOG_ENTRY,
        vec![receiver.clone(), receiver.clone()],
        1,
    );
    assert!(wrong_arity
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATA-HANDLE-ARITY"));
    let wrong_outputs = call(&DATASET_PATH_CATALOG_ENTRY, vec![receiver], 2);
    assert!(!wrong_outputs.diagnostics.is_empty());
}
