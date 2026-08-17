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

    let mut invalid_healing_limit = ExactCadImportOptions::default();
    invalid_healing_limit
        .analysis
        .healing
        .simplify_short_edges_and_sliver_faces = true;
    invalid_healing_limit
        .analysis
        .maximum_healing_displacement_m = 0.0;
    assert!(matches!(
        import_exact_cad(
            "shape.brep",
            b"not dispatched",
            GeometryFormat::Brep,
            &invalid_healing_limit,
            &GeometryImportContext::new(),
        ),
        Err(GeometryImportError::InvalidOptions(reason))
            if reason.contains("positive maximum healing displacement")
    ));

    let mut conflicting_healing = ExactCadImportOptions::default();
    conflicting_healing.analysis.healing.sew = true;
    conflicting_healing
        .analysis
        .healing
        .repair_tolerance_scale_gaps = true;
    assert!(matches!(
        import_exact_cad(
            "shape.brep",
            b"not dispatched",
            GeometryFormat::Brep,
            &conflicting_healing,
            &GeometryImportContext::new(),
        ),
        Err(GeometryImportError::InvalidOptions(reason))
            if reason.contains("separate geometry revisions")
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
    assert!(millimeter_shape
        .topology
        .vertices
        .iter()
        .any(|vertex| vertex.point_m == [0.0, 0.0, 0.0]));
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
#[cfg(all(not(target_arch = "wasm32"), feature = "occt-native"))]
fn occt_orientation_repair_is_explicit_and_preserves_persistent_names() {
    let mut reversed = BOX.to_vec();
    let terminal_orientation = reversed
        .windows(8)
        .rposition(|window| window == b"\n+1 0 *\n")
        .expect("box fixture has a terminal solid orientation");
    reversed[terminal_orientation + 1] = b'-';

    let context = GeometryImportContext::new();
    assert!(import_exact_cad(
        "reversed-box.brep",
        &reversed,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &context,
    )
    .is_err());

    let mut options = ExactCadImportOptions::default();
    options.analysis.healing.repair_orientation = true;
    let healed = import_exact_cad(
        "reversed-box.brep",
        &reversed,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    let canonical = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &context,
    )
    .unwrap();
    assert_eq!(healed.topology, canonical.topology);
    let report = healed
        .healing_report
        .as_ref()
        .expect("changed topology carries healing evidence");
    report.validate().unwrap();
    assert_eq!(report.operations.len(), 1);
    assert_eq!(
        report.operations[0].kind,
        runmat_geometry_core::GeometryHealingOperationKind::RepairOrientation
    );
    assert_eq!(report.operations[0].maximum_displacement_m, 0.0);
    assert!(!report.original_validity.orientation_consistent);
    assert!(report.healed_validity.is_valid());
    assert_eq!(report.revision_map.source_revision.revision, 1);
    assert_eq!(report.revision_map.target_revision.revision, 2);
    assert_eq!(healed.analysis_options().revision.revision, 2);
    let closure = healed.build_closure().unwrap();
    assert!(closure.healing_bytes.is_some());
    assert_eq!(
        closure.manifest.healing_report.as_ref().unwrap().digest,
        runmat_geometry_core::GeometryDigest::from_bytes(
            sha2::Sha256::digest(closure.healing_bytes.as_ref().unwrap()).into()
        )
    );

    let unchanged =
        import_exact_cad("box.brep", BOX, GeometryFormat::Brep, &options, &context).unwrap();
    assert!(unchanged.healing_report.is_none());
    assert_eq!(unchanged.analysis_options().revision.revision, 1);

    let mut identity_limited = options.clone();
    identity_limited.max_identity_work_bytes = 1;
    assert!(matches!(
        import_exact_cad(
            "reversed-box.brep",
            &reversed,
            GeometryFormat::Brep,
            &identity_limited,
            &context,
        ),
        Err(GeometryImportError::ExactValidationBudgetExceeded(reason))
            if reason.contains("persistent identity serialization")
    ));

    let mut exhausted_revision = options;
    exhausted_revision.analysis.revision.revision = u64::MAX;
    assert!(matches!(
        import_exact_cad(
            "reversed-box.brep",
            &reversed,
            GeometryFormat::Brep,
            &exhausted_revision,
            &context,
        ),
        Err(GeometryImportError::InvalidOptions(reason))
            if reason.contains("cannot advance")
    ));
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
