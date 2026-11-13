use super::*;

#[test]
fn patch_descriptor_keeps_coordinate_and_face_vertex_forms() {
    let labels: Vec<_> = PATCH_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"h = patch(X, Y, C)"));
    assert!(labels.contains(&"h = patch(ax, ...)"));
    assert!(PATCH_CATALOG_ENTRY
        .integer_capabilities
        .iter()
        .any(|capability| { capability.form.contains("integer_CData") }));
}
