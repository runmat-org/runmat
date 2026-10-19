use super::*;
use runmat_types::{
    CallRequest, LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

#[test]
fn region_projection_has_scalar_struct_fact_and_typed_receiver() {
    let asset = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::super::load::GEOMETRY_ASSET_CLASS.owned()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    let request = |value| CallRequest {
        arguments: vec![value],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let valid = inference::infer(&request(asset), &GEOMETRY_LIST_REGIONS_CATALOG_ENTRY);
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    let ValueKindFact::Struct(fields) = &valid.outputs[0].kind else {
        panic!("expected struct")
    };
    assert!(fields.fields.contains_key("regions"));
    let wrong = inference::infer(
        &request(ValueFact::scalar(ValueKindFact::Logical)),
        &GEOMETRY_LIST_REGIONS_CATALOG_ENTRY,
    );
    assert!(wrong
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-GEOMETRY-LISTREGIONS-ASSET"));
}
