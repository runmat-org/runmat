use super::*;
use crate::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

use super::super::{
    MESHING_BOUNDARY_FACES_CATALOG_ENTRY, MESHING_ELEMENTS_CATALOG_ENTRY,
    MESHING_EVIDENCE_CATALOG_ENTRY, MESHING_GENERATE_CATALOG_ENTRY, MESHING_LOAD_CATALOG_ENTRY,
    MESHING_MESH_METHOD_CATALOG_ENTRY, MESHING_NODES_CATALOG_ENTRY, MESHING_QUALITY_CATALOG_ENTRY,
    MESHING_REGIONS_CATALOG_ENTRY, MESHING_SAVE_CATALOG_ENTRY,
};

const ALL: [&BuiltinCatalogEntry; 10] = [
    &MESHING_GENERATE_CATALOG_ENTRY,
    &MESHING_LOAD_CATALOG_ENTRY,
    &MESHING_SAVE_CATALOG_ENTRY,
    &MESHING_NODES_CATALOG_ENTRY,
    &MESHING_ELEMENTS_CATALOG_ENTRY,
    &MESHING_BOUNDARY_FACES_CATALOG_ENTRY,
    &MESHING_REGIONS_CATALOG_ENTRY,
    &MESHING_QUALITY_CATALOG_ENTRY,
    &MESHING_EVIDENCE_CATALOG_ENTRY,
    &MESHING_MESH_METHOD_CATALOG_ENTRY,
];

fn mesh() -> ValueFact {
    host_object(MESHING_MESH_OBJECT_CLASS)
}

fn asset() -> ValueFact {
    host_object(crate::GEOMETRY_ASSET_CLASS)
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> runmat_types::CallInference {
    let BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!("identity inference expected");
    };
    rule.infer(request, entry)
}

fn codes(entry: &BuiltinCatalogEntry, request: &CallRequest) -> Vec<String> {
    infer(entry, request)
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.code)
        .collect()
}

fn output_class(entry: &BuiltinCatalogEntry, request: &CallRequest) -> Option<String> {
    match &infer(entry, request).outputs.first()?.kind {
        ValueKindFact::Object(object) => object.runtime_class.as_ref().map(ToString::to_string),
        _ => None,
    }
}

fn literal(text: &str) -> LiteralValue {
    LiteralValue::String(text.into())
}

#[test]
fn meshing_entries_are_valid_catalog_content() {
    let errors = crate::validate_builtin_catalog(&ALL);
    assert!(errors.is_empty(), "{errors:?}");
    for entry in ALL {
        assert!(entry
            .documentation
            .sections
            .iter()
            .any(|section| section.heading == "Use in a study"));
        assert!(!entry.documentation.examples.is_empty());
    }
    assert_eq!(
        MESHING_METHOD_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::MethodOnly
    );
    assert_eq!(
        MESHING_GENERATE_CATALOG_ENTRY.placement.portability,
        BuiltinPortability::NativeOnly
    );
}

#[test]
fn generate_returns_a_mesh_and_checks_options() {
    let valid = request(
        vec![asset(), text(), unknown(), text(), text()],
        vec![
            LiteralValue::Unknown,
            literal("TargetSize"),
            LiteralValue::Number(0.005),
            literal("Element_Order"),
            literal("TET10"),
        ],
    );
    assert!(codes(&MESHING_GENERATE_CATALOG_ENTRY, &valid).is_empty());
    assert_eq!(
        output_class(&MESHING_GENERATE_CATALOG_ENTRY, &valid).as_deref(),
        Some("meshing.Mesh")
    );
    let bad_order = request(
        vec![asset(), text(), text()],
        vec![
            LiteralValue::Unknown,
            literal("ElementOrder"),
            literal("hex8"),
        ],
    );
    assert!(codes(&MESHING_GENERATE_CATALOG_ENTRY, &bad_order)
        .contains(&"RM-CATALOG-MESHING-GENERATE-ORDER".to_string()));
    let bad_option = request(
        vec![asset(), text(), unknown()],
        vec![LiteralValue::Unknown, literal("Material")],
    );
    assert!(codes(&MESHING_GENERATE_CATALOG_ENTRY, &bad_option)
        .contains(&"RM-CATALOG-MESHING-GENERATE-OPTION".to_string()));
    assert!(codes(
        &MESHING_GENERATE_CATALOG_ENTRY,
        &request(vec![asset(), text()], vec![])
    )
    .contains(&"RM-CATALOG-MESHING-GENERATE-OPTIONS".to_string()));
    assert!(
        codes(&MESHING_GENERATE_CATALOG_ENTRY, &request(vec![], vec![]))
            .contains(&"RM-CATALOG-MESHING-GENERATE-ARITY".to_string())
    );
    assert!(codes(
        &MESHING_GENERATE_CATALOG_ENTRY,
        &request(vec![mesh()], vec![])
    )
    .contains(&"RM-CATALOG-MESHING-GENERATE-GEOMETRY".to_string()));
}

#[test]
fn load_and_save_check_paths_and_force() {
    let load = request(vec![text(), text()], vec![]);
    assert!(codes(&MESHING_LOAD_CATALOG_ENTRY, &load).is_empty());
    assert_eq!(
        output_class(&MESHING_LOAD_CATALOG_ENTRY, &load).as_deref(),
        Some("meshing.Mesh")
    );
    assert!(
        codes(&MESHING_LOAD_CATALOG_ENTRY, &request(vec![text()], vec![]))
            .contains(&"RM-CATALOG-MESHING-LOAD-ARITY".to_string())
    );
    let numeric = ValueFact::scalar(ValueKindFact::Logical);
    assert!(codes(
        &MESHING_LOAD_CATALOG_ENTRY,
        &request(vec![text(), numeric], vec![])
    )
    .contains(&"RM-CATALOG-MESHING-LOAD-PATH".to_string()));

    let save = request(
        vec![
            mesh(),
            text(),
            text(),
            text(),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            literal("a.cbor"),
            literal("b.cbor"),
            literal("Force"),
        ],
    );
    assert!(codes(&MESHING_SAVE_CATALOG_ENTRY, &save).is_empty());
    let same = request(
        vec![mesh(), text(), text()],
        vec![LiteralValue::Unknown, literal("a.cbor"), literal("a.cbor")],
    );
    assert!(codes(&MESHING_SAVE_CATALOG_ENTRY, &same)
        .contains(&"RM-CATALOG-MESHING-SAVE-PATHS".to_string()));
    let numeric_force = request(
        vec![
            mesh(),
            text(),
            text(),
            text(),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![],
    );
    assert!(codes(&MESHING_SAVE_CATALOG_ENTRY, &numeric_force)
        .contains(&"RM-CATALOG-MESHING-SAVE-FORCE".to_string()));
    assert!(codes(
        &MESHING_SAVE_CATALOG_ENTRY,
        &request(vec![asset(), text(), text()], vec![])
    )
    .contains(&"RM-CATALOG-MESHING-SAVE-MESH".to_string()));
}

#[test]
fn inspections_require_one_mesh_and_accept_unknown_values() {
    for entry in [
        &MESHING_NODES_CATALOG_ENTRY,
        &MESHING_ELEMENTS_CATALOG_ENTRY,
        &MESHING_BOUNDARY_FACES_CATALOG_ENTRY,
        &MESHING_REGIONS_CATALOG_ENTRY,
        &MESHING_QUALITY_CATALOG_ENTRY,
        &MESHING_EVIDENCE_CATALOG_ENTRY,
    ] {
        assert!(
            codes(entry, &request(vec![mesh()], vec![])).is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(
            codes(entry, &request(vec![unknown()], vec![])).is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(
            !codes(entry, &request(vec![text()], vec![])).is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(
            !codes(entry, &request(vec![mesh(), mesh()], vec![])).is_empty(),
            "{}",
            entry.identity.name
        );
    }
    assert_eq!(
        output_class(
            &MESHING_EVIDENCE_CATALOG_ENTRY,
            &request(vec![mesh()], vec![])
        )
        .as_deref(),
        Some("meshing.Evidence")
    );
}

#[test]
fn display_checks_option_names_and_values() {
    let valid = request(
        vec![mesh(), text(), text(), text(), unknown()],
        vec![
            LiteralValue::Unknown,
            literal("Regions"),
            literal("solid-region:abc"),
            literal("Section"),
        ],
    );
    assert!(codes(&MESHING_MESH_METHOD_CATALOG_ENTRY, &valid).is_empty());
    let bad = request(
        vec![mesh(), text(), text()],
        vec![LiteralValue::Unknown, literal("Edges"), literal("on")],
    );
    assert!(codes(&MESHING_MESH_METHOD_CATALOG_ENTRY, &bad)
        .contains(&"RM-CATALOG-MESHING-MESH-VALUE".to_string()));
    let unknown_option = request(
        vec![mesh(), text(), unknown()],
        vec![LiteralValue::Unknown, literal("XRay")],
    );
    assert!(codes(&MESHING_MESH_METHOD_CATALOG_ENTRY, &unknown_option)
        .contains(&"RM-CATALOG-MESHING-MESH-OPTION".to_string()));
}
