use super::*;
use runmat_types::{
    CallRequest, LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[test]
fn meshes_prove_cell_and_patch_fields_without_guessing_surface_count() {
    let asset = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::super::load::GEOMETRY_ASSET_CLASS.owned()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    let request = CallRequest {
        arguments: vec![asset],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &GEOMETRY_MESHES_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("expected cell")
    };
    let ValueKindFact::Struct(mesh) = &cell.element.kind else {
        panic!("expected mesh struct")
    };
    assert_eq!(
        mesh.fields["vertices"].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert_eq!(
        mesh.fields["faces"].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
}
