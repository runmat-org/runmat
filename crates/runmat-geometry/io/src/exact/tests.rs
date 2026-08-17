use super::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
use runmat_geometry_core::{ExactMassPropertiesImplementation, TopologicalOrientation};

const BOX: &[u8] = include_bytes!("../../tests/fixtures/box.brep");

#[test]
fn exact_analysis_options_are_validated_before_kernel_dispatch() {
    let mut invalid_tolerance = ExactCadImportOptions::default();
    invalid_tolerance.analysis.requested_deviation_m = 0.0;
    assert!(matches!(
        import_exact_cad(
            "shape.brep",
            b"not dispatched",
            GeometryFormat::Brep,
            &invalid_tolerance,
            &GeometryImportContext::new(),
        ),
        Err(GeometryImportError::InvalidOptions(reason))
            if reason.contains("requested_deviation_m")
    ));

    let mut invalid_revision = ExactCadImportOptions::default();
    invalid_revision.analysis.revision.revision = 0;
    assert!(matches!(
        import_exact_cad(
            "shape.brep",
            b"not dispatched",
            GeometryFormat::Brep,
            &invalid_revision,
            &GeometryImportContext::new(),
        ),
        Err(GeometryImportError::InvalidOptions(reason))
            if reason.contains("revision")
    ));
}

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
    let closure = first.build_closure().unwrap();
    let renamed_closure = second.build_closure().unwrap();
    assert_eq!(closure, renamed_closure);
    assert_eq!(first.analysis_options(), &options.analysis);
    assert_eq!(closure.document.revision, options.analysis.revision);
    assert_eq!(closure.document.healing, options.analysis.healing);
    assert_eq!(
        closure.document.tolerance.maximum_healing_displacement_m,
        options.analysis.maximum_healing_displacement_m
    );
    assert_eq!(closure.document.source.content_digest, first.source_digest);
    assert_eq!(closure.document.source.format, GeometrySourceFormat::Brep);
    assert_eq!(closure.document.source.source_units, UnitSystem::Meter);
    assert_eq!(
        closure
            .manifest
            .kernel_representation
            .as_ref()
            .unwrap()
            .digest,
        GeometryDigest::from_bytes(first.representation_digest())
    );
    assert_eq!(first.topology.assemblies.len(), 1);
    assert_eq!(first.topology.bodies.len(), 1);
    assert_eq!(first.topology.lumps.len(), 1);
    assert!(first.topology.lumps[0]
        .id
        .source_topology_id
        .starts_with("occt:"));
    assert_eq!(first.topology.lumps[0].id.source_topology_id.len(), 69);
    assert_eq!(first.topology.lumps[0].solid_ids.len(), 1);
    assert_eq!(
        first.topology.lumps[0].solid_ids[0],
        first.topology.solids[0].id
    );
    assert_eq!(first.topology.solids.len(), 1);
    assert_eq!(first.topology.shells.len(), 1);
    assert_eq!(first.topology.faces.len(), 6);
    assert_eq!(first.topology.wires.len(), 6);
    assert_eq!(first.topology.coedges.len(), 24);
    assert_eq!(first.topology.edges.len(), 12);
    assert_eq!(first.topology.vertices.len(), 8);
    assert_eq!(first.evaluators.curves.len(), 12);
    assert_eq!(first.evaluators.pcurves.len(), 24);
    assert_eq!(first.evaluators.surfaces.len(), 6);
    assert_eq!(first.evaluators.trim_classifiers.len(), 6);
    assert_eq!(first.evaluators.mass_properties.len(), 1);
    first.topology.validate_solid_shell_boundaries().unwrap();
    assert!(first
        .evaluators
        .kernel_representation_digest()
        .unwrap()
        .is_some());
    assert!(first.evaluators.kernel_abi.starts_with("occt/"));
    assert!(first
        .representation
        .windows(16)
        .any(|window| window == b"Triangulations 0"));
    let ExactMassPropertiesImplementation::KernelValidated {
        properties: mass, ..
    } = first.evaluators.mass_properties[0].implementation
    else {
        panic!("solid import must contain kernel-validated mass properties");
    };
    assert!((mass.volume_m3 - 6.0).abs() < 1.0e-12);
    assert!((mass.surface_area_m2 - 22.0).abs() < 1.0e-12);
    assert_eq!(mass.centroid_m, [0.5, 1.0, 1.5]);

    let mut non_manifold = first.topology.clone();
    non_manifold.coedges[0].edge_id = non_manifold.coedges[1].edge_id.clone();
    assert!(non_manifold.validate_solid_shell_boundaries().is_err());

    let mut misoriented = first.topology.clone();
    let shared_edge = misoriented.coedges[0].edge_id.clone();
    let matching_use = misoriented
        .coedges
        .iter()
        .enumerate()
        .find(|(index, coedge)| *index != 0 && coedge.edge_id == shared_edge)
        .map(|(index, _)| index)
        .unwrap();
    misoriented.coedges[matching_use].orientation =
        match misoriented.coedges[matching_use].orientation {
            TopologicalOrientation::Forward => TopologicalOrientation::Reversed,
            TopologicalOrientation::Reversed => TopologicalOrientation::Forward,
        };
    assert!(misoriented.validate_solid_shell_boundaries().is_err());

    let mut millimeter_options = options.clone();
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
    assert_eq!(millimeter_shape.topology.vertices[0].point_m[0], 0.0);
    assert!(millimeter_shape
        .topology
        .vertices
        .iter()
        .flat_map(|vertex| vertex.point_m)
        .all(|coordinate| coordinate <= 0.003));
    let ExactMassPropertiesImplementation::KernelValidated {
        properties: millimeter_mass,
        ..
    } = millimeter_shape.evaluators.mass_properties[0].implementation
    else {
        panic!("solid import must contain kernel-validated mass properties");
    };
    assert!((millimeter_mass.volume_m3 - 6.0e-9).abs() < 1.0e-20);
    assert_eq!(millimeter_mass.centroid_m, [0.0005, 0.001, 0.0015]);

    let mut byte_limited = options.clone();
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
    let mut entity_limited = options.clone();
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
    let mut validation_limited = options;
    validation_limited.max_validation_search_work = 1;
    assert!(matches!(
        import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &validation_limited,
            &context,
        ),
        Err(GeometryImportError::ExactValidationBudgetExceeded(_))
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
