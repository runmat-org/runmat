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
fn accessor_identities_are_unique_sorted_and_source_linked() {
    let names: Vec<_> = ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(
        names,
        [
            "Dataset.array",
            "Dataset.arrays",
            "Dataset.has_array",
            "Dataset.id",
            "Dataset.path",
            "Dataset.version"
        ]
    );
    assert!(ENTRIES.iter().all(|entry| !entry.bindings.is_empty()));
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:#?}");
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
