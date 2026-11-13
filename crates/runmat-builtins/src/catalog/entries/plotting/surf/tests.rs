use super::*;

#[test]
fn surface_descriptor_keeps_geometry_and_independent_color_forms() {
    let labels: Vec<_> = SURF_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    for label in [
        "h = surf(Z)",
        "h = surf(X, Y, Z)",
        "h = surf(Z, C)",
        "h = surf(X, Y, Z, C)",
    ] {
        assert!(labels.contains(&label));
    }
}
