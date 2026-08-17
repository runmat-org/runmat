use super::*;

const BOX: &[u8] = include_bytes!("../../tests/fixtures/box.brep");

#[test]
#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
fn occt_import_is_non_tessellating_bounded_and_deterministic() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    let options = ExactCadImportOptions::default();
    let context = GeometryImportContext::new();
    let first =
        import_exact_cad("box.brep", BOX, GeometryFormat::Brep, &options, &context).unwrap();
    let second = import_exact_cad(
        "renamed-box.brep",
        BOX,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.topology.solid_count, 1);
    assert_eq!(first.topology.shell_count, 1);
    assert_eq!(first.topology.face_count, 6);
    assert_eq!(first.topology.wire_count, 6);
    assert_eq!(first.topology.edge_count, 12);
    assert_eq!(first.topology.vertex_count, 8);
    assert!(first.kernel_abi.starts_with("occt/"));
    assert!(first
        .representation
        .windows(16)
        .any(|window| window == b"Triangulations 0"));
    let mass = first.mass_properties.unwrap();
    assert!((mass.volume_m3 - 6.0).abs() < 1.0e-12);
    assert!((mass.surface_area_m2 - 22.0).abs() < 1.0e-12);
    assert_eq!(mass.centroid_m, [0.5, 1.0, 1.5]);

    let mut millimeter_options = options;
    millimeter_options.source_units = UnitSystem::Millimeter;
    let millimeter_shape = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &millimeter_options,
        &context,
    )
    .unwrap();
    assert_eq!(millimeter_shape.representation, first.representation);
    let millimeter_mass = millimeter_shape.mass_properties.unwrap();
    assert!((millimeter_mass.volume_m3 - 6.0e-9).abs() < 1.0e-20);
    assert_eq!(millimeter_mass.centroid_m, [0.0005, 0.001, 0.0015]);

    let mut byte_limited = options;
    byte_limited.max_representation_bytes = 1;
    assert!(matches!(
        import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &byte_limited,
            &context,
        ),
        Err(GeometryImportError::ExactRepresentationCapacityExceeded { limit: 1 })
    ));
    let mut entity_limited = options;
    entity_limited.max_entities = 5;
    assert!(matches!(
        import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &entity_limited,
            &context,
        ),
        Err(GeometryImportError::ExactEntityCapacityExceeded { limit: 5 })
    ));

    let cancelled = Arc::new(AtomicBool::new(true));
    assert!(matches!(
        import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &ExactCadImportOptions::default(),
            &GeometryImportContext::with_cancellation(cancelled.clone()),
        ),
        Err(GeometryImportError::Cancelled)
    ));
    assert!(cancelled.load(Ordering::Relaxed));
}

#[test]
#[cfg(not(all(not(target_arch = "wasm32"), feature = "occt-native")))]
fn import_is_capability_honest_without_a_native_kernel() {
    let error = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap_err();
    assert!(matches!(error, GeometryImportError::BackendUnavailable(_)));
}
