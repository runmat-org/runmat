use super::*;
use crate::catalog::inference::infer_catalog_call;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn call(entry: &BuiltinCatalogEntry, args: Vec<ValueFact>, count: usize) -> CallInference {
    infer_catalog_call(
        entry,
        &CallRequest {
            literals: LiteralContext::default(),
            arguments: args,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
    )
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn schema() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([(
            "arrays".into(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                BTreeMap::new(),
                true,
            ))),
        )]),
        true,
    )))
}

#[test]
fn all_ten_path_identities_are_distinct_and_sorted() {
    let names: Vec<_> = ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(names.len(), 10);
    assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(ENTRIES.iter().all(|entry| !entry.bindings.is_empty()));
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn typed_dataset_outputs_and_integer_schema_contract_match_runtime() {
    let created = call(&DATA_CREATE_CATALOG_ENTRY, vec![text(), schema()], 1);
    assert!(created.diagnostics.is_empty(), "{:?}", created.diagnostics);
    let ValueKindFact::Object(object) = &created.outputs[0].kind else {
        panic!("data.create must infer a Dataset object")
    };
    assert!(object
        .runtime_class
        .as_ref()
        .is_some_and(|name| name.is(DATASET_CLASS)));
    assert!(object.properties.contains_key("__data_path"));
    assert!(created.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(created
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert_eq!(DATA_CREATE_CATALOG_ENTRY.integer_capabilities.len(), 1);

    let opened = call(&DATA_OPEN_CATALOG_ENTRY, vec![text()], 1);
    assert!(matches!(opened.outputs[0].kind, ValueKindFact::Object(_)));
    assert!(opened.effects.0.contains(&EffectKind::FilesystemRead));
}

#[test]
fn known_invalid_path_schema_format_and_output_count_are_diagnosed() {
    let wrong_path = call(
        &DATA_COPY_CATALOG_ENTRY,
        vec![ValueFact::scalar(ValueKindFact::Logical), text()],
        1,
    );
    assert!(wrong_path
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-PATH-TEXT"));

    let missing_arrays = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        true,
    )));
    let wrong_schema = call(&DATA_CREATE_CATALOG_ENTRY, vec![text(), missing_arrays], 1);
    assert!(wrong_schema
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-SCHEMA"));

    let mut bad_format = CallRequest {
        arguments: vec![text(), text(), text()],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("csv".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = infer_catalog_call(&DATA_IMPORT_CATALOG_ENTRY, &bad_format);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-FORMAT"));
    bad_format.arguments[1] = ValueFact::scalar(ValueKindFact::Unknown);
    bad_format.literals = LiteralContext::default();
    assert!(infer_catalog_call(&DATA_IMPORT_CATALOG_ENTRY, &bad_format)
        .diagnostics
        .is_empty());

    let excess_outputs = call(&DATA_INSPECT_CATALOG_ENTRY, vec![text()], 2);
    assert!(!excess_outputs.diagnostics.is_empty());

    let singleton_text_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(
        call(&DATA_OPEN_CATALOG_ENTRY, vec![singleton_text_array], 1)
            .diagnostics
            .is_empty()
    );
    let paged_char = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(1), Some(2)]),
        StorageFact::Dense,
    );
    assert!(call(&DATA_OPEN_CATALOG_ENTRY, vec![paged_char], 1)
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-PATH-TEXT"));
}

#[test]
fn list_inspect_and_logical_results_have_precise_host_facts() {
    let list = call(&DATA_LIST_CATALOG_ENTRY, vec![text()], 1);
    assert!(matches!(list.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(list.outputs[0].residency, ResidencyFact::Host);

    let inspect = call(&DATA_INSPECT_CATALOG_ENTRY, vec![text()], 1);
    let ValueKindFact::Struct(fields) = &inspect.outputs[0].kind else {
        panic!("inspect structure")
    };
    for field in ["path", "id", "arrayCount", "updatedAt"] {
        assert!(fields.fields.contains_key(field));
    }

    for entry in [
        &DATA_EXISTS_CATALOG_ENTRY,
        &DATA_DELETE_CATALOG_ENTRY,
        &DATA_COPY_CATALOG_ENTRY,
        &DATA_MOVE_CATALOG_ENTRY,
        &DATA_EXPORT_CATALOG_ENTRY,
    ] {
        let args = match entry.identity.name {
            "data.copy" | "data.move" => vec![text(), text()],
            "data.export" => vec![text(), text(), text()],
            _ => vec![text()],
        };
        let inferred = call(entry, args, 1);
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Logical));
    }
}

#[test]
fn move_effects_and_shared_admission_failures_match_runtime() {
    let moved = call(&DATA_MOVE_CATALOG_ENTRY, vec![text(), text()], 1);
    assert!(moved.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(moved.effects.0.contains(&EffectKind::FilesystemWrite));

    let expected = [
        "RUNMAT:Data:Error",
        "RunMat:Placement:InsufficientResourceEvidence",
        "RunMat:Placement:StaleResourceAuthority",
        "RunMat:Placement:ResourceLimit",
        "RunMat:Placement:Unavailable",
        "RunMat:Placement:InvalidRevision",
        "RunMat:Placement:InvalidDecision",
        "RunMat:Cancelled",
    ];
    for entry in ENTRIES {
        let identifiers: Vec<_> = entry
            .descriptor
            .errors
            .iter()
            .filter_map(|error| error.identifier)
            .collect();
        assert_eq!(identifiers, expected, "{}", entry.identity.name);
    }
}
