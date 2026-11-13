use super::CTRANSPOSE_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{
    AliasFact, CallRequest, CellFact, EffectKind, LayoutFact, LiteralContext, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact,
    StorageFact, StructFact, ValueFact, ValueKindFact, ViewFact,
};
use std::collections::BTreeMap;

fn infer(input: ValueFact) -> runmat_types::CallInference {
    infer_catalog_call(
        &CTRANSPOSE_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

fn numeric(shape: ShapeFact, storage: StorageFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    )
}

#[test]
fn numeric_transpose_preserves_sparse_and_does_not_promise_device_residency_or_uniqueness() {
    let sparse = numeric(ShapeFact::from(vec![Some(2), Some(3)]), StorageFact::Sparse);
    let host = infer(sparse);
    assert_eq!(
        host.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    assert_eq!(host.outputs[0].storage, StorageFact::Sparse);
    assert_eq!(host.outputs[0].residency, ResidencyFact::Host);
    assert!(!host.effects.0.contains(&EffectKind::HostCallback));

    let singleton = infer(numeric(
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Sparse,
    ));
    assert_eq!(singleton.outputs[0].storage, StorageFact::Sparse);
    assert_eq!(
        singleton.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );

    let mut device = numeric(ShapeFact::from(vec![Some(2), Some(3)]), StorageFact::Dense);
    device.residency = ResidencyFact::Device {
        provider: Some("exact".into()),
    };
    let resident = infer(device);
    assert_eq!(
        resident.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    assert_eq!(resident.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(resident.outputs[0].alias, AliasFact::Unknown);
    assert_eq!(resident.outputs[0].view, ViewFact::Unknown);
    assert_eq!(resident.outputs[0].layout, LayoutFact::Unknown);
}

#[test]
fn container_transpose_changes_shape_without_reusing_old_positional_facts() {
    let element = numeric(ShapeFact::Scalar, StorageFact::Scalar);
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element.clone()),
            elements: vec![element; 6],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Opaque,
    );
    let result = infer(cell);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    let ValueKindFact::Cell(fact) = &result.outputs[0].kind else {
        panic!("cell class changed");
    };
    assert!(fact.elements.is_empty());
    assert!(!fact.elements_complete);
    assert!(matches!(&fact.element.kind, ValueKindFact::Numeric(_)));

    let mut fields = BTreeMap::new();
    fields.insert(
        "id".to_string(),
        numeric(ShapeFact::Scalar, StorageFact::Scalar),
    );
    let structure = ValueFact::proven(
        ValueKindFact::Struct(StructFact::array(
            fields.clone(),
            true,
            vec![fields; 6],
            true,
        )),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Opaque,
    );
    let result = infer(structure);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    let ValueKindFact::Struct(fact) = &result.outputs[0].kind else {
        panic!("struct class changed");
    };
    assert!(fact.elements.is_empty());
    assert!(!fact.elements_complete);
    assert!(fact.fields.contains_key("id"));

    let string = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert_eq!(
        infer(string).outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );

    let invalid = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(2), Some(3), Some(2)]),
        StorageFact::Dense,
    );
    assert!(infer(invalid)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-MATRIX-RANK"));
}

#[test]
fn scalar_struct_identity_keeps_alias_fact_and_unknown_receiver_keeps_callback_effects() {
    let mut structure = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(BTreeMap::new(), true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    structure.alias = AliasFact::Shared;
    structure.view = ViewFact::ReadOnlyView;
    let result = infer(structure);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].alias, AliasFact::Shared);
    assert_eq!(result.outputs[0].view, ViewFact::ReadOnlyView);
    assert!(!result.effects.0.contains(&EffectKind::HostCallback));

    let mut text = ValueFact::scalar(ValueKindFact::String);
    text.alias = AliasFact::Shared;
    text.view = ViewFact::ReadOnlyView;
    let result = infer(text);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].alias, AliasFact::Shared);
    assert_eq!(result.outputs[0].view, ViewFact::ReadOnlyView);

    let unknown = infer(ValueFact::unknown(
        runmat_types::DynamicReason::RuntimeValue,
    ));
    assert!(unknown.effects.0.contains(&EffectKind::HostCallback));
    assert!(unknown.effects.0.contains(&EffectKind::Unknown));
}

#[test]
fn singleton_container_arrays_do_not_inherit_receiver_alias_or_view() {
    let mut cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::String)),
            elements: vec![ValueFact::scalar(ValueKindFact::String)],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    cell.alias = AliasFact::Shared;
    cell.view = ViewFact::ReadOnlyView;
    let result = infer(cell);
    assert_eq!(result.outputs[0].alias, AliasFact::Unique);
    assert_eq!(result.outputs[0].view, ViewFact::Materialized);

    let mut text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    text.alias = AliasFact::Shared;
    text.view = ViewFact::ReadOnlyView;
    let result = infer(text);
    assert_eq!(result.outputs[0].alias, AliasFact::Unique);
    assert_eq!(result.outputs[0].view, ViewFact::Materialized);

    let mut structure = ValueFact::proven(
        ValueKindFact::Struct(StructFact::array(
            BTreeMap::new(),
            true,
            vec![BTreeMap::new()],
            true,
        )),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    structure.alias = AliasFact::Shared;
    structure.view = ViewFact::ReadOnlyView;
    let result = infer(structure);
    assert_eq!(result.outputs[0].alias, AliasFact::Unique);
    assert_eq!(result.outputs[0].view, ViewFact::Materialized);
}

#[test]
fn canonical_documentation_retains_legacy_reference_inventory() {
    // Immutable test input conserved from 52863699e3's retired sidecar,
    // SHA-256 c4d33e2c326c506ff4325004f68c46039fbf90736db1889b35911604fa3d33d0.
    // Catalog documentation remains the sole production authoring source.
    let legacy: serde_json::Value =
        serde_json::from_str(include_str!("tests/fixtures/ctranspose_52863699e3.json"))
            .expect("legacy ctranspose reference JSON");
    let documentation = &CTRANSPOSE_CATALOG_ENTRY.documentation;
    assert_eq!(
        documentation.examples.len(),
        legacy["examples"].as_array().unwrap().len()
    );
    assert_eq!(
        documentation.faqs.len(),
        legacy["faqs"].as_array().unwrap().len()
    );
    assert_eq!(
        documentation.links.len(),
        legacy["links"].as_array().unwrap().len() + 1
    );
    assert_eq!(
        documentation.keywords.len(),
        legacy["keywords"].as_array().unwrap().len()
    );
    assert_eq!(documentation.sections.len(), 2);
    assert!(documentation.status.is_none());
    for (legacy, canonical) in legacy["examples"]
        .as_array()
        .unwrap()
        .iter()
        .zip(documentation.examples)
    {
        assert_eq!(canonical.program, legacy["input"].as_str().unwrap());
        assert_eq!(canonical.display_output, legacy["output"].as_str());
    }
    for (legacy, canonical) in legacy["examples"]
        .as_array()
        .unwrap()
        .iter()
        .zip(documentation.examples)
        .take(5)
    {
        assert_eq!(canonical.title, legacy["description"].as_str().unwrap());
    }
    for (legacy, canonical) in legacy["faqs"]
        .as_array()
        .unwrap()
        .iter()
        .zip(documentation.faqs)
    {
        assert_eq!(canonical.question, legacy["question"].as_str().unwrap());
    }
    for index in [0, 1, 2, 5, 6, 7] {
        assert_eq!(
            documentation.faqs[index].answer,
            legacy["faqs"][index]["answer"].as_str().unwrap()
        );
    }
    for (legacy, canonical) in legacy["links"]
        .as_array()
        .unwrap()
        .iter()
        .zip(documentation.links)
    {
        assert_eq!(canonical.label, legacy["label"].as_str().unwrap());
    }
    for (legacy, canonical) in legacy["keywords"]
        .as_array()
        .unwrap()
        .iter()
        .zip(documentation.keywords)
    {
        assert_eq!(*canonical, legacy.as_str().unwrap());
    }
    assert!(documentation.sections[1]
        .paragraphs
        .iter()
        .any(|paragraph| paragraph.contains("re-upload")));
    assert_eq!(
        documentation.description,
        legacy["description"].as_str().unwrap()
    );
    assert_eq!(documentation.summary, legacy["summary"].as_str().unwrap());
    assert!(documentation.links.iter().any(|link| {
        matches!(link.target, crate::BuiltinDocumentationLinkTarget::Source(url)
            if url == legacy["source"]["url"].as_str().unwrap())
    }));
}
