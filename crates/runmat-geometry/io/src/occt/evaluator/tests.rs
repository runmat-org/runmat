use super::*;
use crate::import::LabeledSubshapeRemapConflictKind;
use crate::{
    import::{GeometryImportContext, GeometryImportError},
    import_exact_cad, ExactCadImportOptions, GeometryFormat,
};
use runmat_geometry_core::{
    ExactContactDefinition, GeometryEvaluationError, GeometryEvaluationErrorKind,
    GeometryHealingOperationKind, MassPropertiesEvaluatorId, SurfaceEvaluatorId, TrimClassifierId,
    TrimDomainLocation, UnitSystem,
};
use sha2::Digest as _;

const BOX: &[u8] = include_bytes!("../../../tests/fixtures/box.brep");
const BOX_CAVITY: &[u8] = include_bytes!("../../../tests/fixtures/box_cavity.brep");
const COINCIDENT_SOLIDS: &[u8] = include_bytes!("../../../tests/fixtures/coincident_solids.brep");
const DISCONNECTED_SOLIDS: &[u8] =
    include_bytes!("../../../tests/fixtures/disconnected_solids.brep");
const DISCONNECTED_SOLIDS_REVERSED: &[u8] =
    include_bytes!("../../../tests/fixtures/disconnected_solids_reversed.brep");
const GAPPED_SHEET: &[u8] = include_bytes!("../../../tests/fixtures/gapped_sheet.brep");
const INDEPENDENT_CONTACT: &[u8] =
    include_bytes!("../../../tests/fixtures/independent_contact.step");
const INVALID_CAVITY: &[u8] = include_bytes!("../../../tests/fixtures/invalid_cavity.brep");
const MIXED_SOLID_SHEET: &[u8] = include_bytes!("../../../tests/fixtures/mixed_solid_sheet.brep");
const SHORT_EDGE_FACE: &[u8] = include_bytes!("../../../tests/fixtures/short_edge_face.brep");
const SHARED_INTERFACE: &[u8] = include_bytes!("../../../tests/fixtures/shared_interface.brep");
const SLIVER_FACE_SHEET: &[u8] = include_bytes!("../../../tests/fixtures/sliver_face_sheet.brep");
const TWO_BOX_ASSEMBLY: &[u8] = include_bytes!("../../../tests/fixtures/two_box_assembly.step");
const XCAF_SHORT_EDGE_ASSEMBLY: &[u8] =
    include_bytes!("../../../tests/fixtures/xcaf_short_edge_assembly.step");
const XCAF_LABELED_FACE_ASSEMBLY: &[u8] =
    include_bytes!("../../../tests/fixtures/xcaf_labeled_face_assembly.step");
const XCAF_LABELED_SHORT_EDGE_ASSEMBLY: &[u8] =
    include_bytes!("../../../tests/fixtures/xcaf_labeled_short_edge_assembly.step");

struct Unlimited;

impl GeometryEvaluationControl for Unlimited {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

#[test]
fn exact_evaluator_reconstructs_from_the_transferable_closure() {
    let imported = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let direct = OcctExactEvaluator::new(&imported).unwrap();
    let reconstructed = OcctExactEvaluator::from_closure(
        &imported.representation,
        imported.meters_per_source_unit,
        &imported.topology,
        &imported.evaluators,
    )
    .unwrap();
    let curve_id = &imported.topology.edges[0].curve_evaluator_id;
    let range = direct.parameter_range(curve_id).unwrap();
    assert_eq!(
        direct.point(curve_id, range.start, &Unlimited).unwrap(),
        reconstructed
            .point(curve_id, range.start, &Unlimited)
            .unwrap()
    );

    let mut poisoned_representation = imported.representation.clone();
    poisoned_representation[0] ^= 1;
    assert_eq!(
        OcctExactEvaluator::from_closure(
            &poisoned_representation,
            imported.meters_per_source_unit,
            &imported.topology,
            &imported.evaluators,
        )
        .err()
        .unwrap()
        .kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
    assert!(OcctExactEvaluator::from_closure(
        &imported.representation,
        0.0,
        &imported.topology,
        &imported.evaluators,
    )
    .is_err());
}

#[test]
fn exact_import_maps_every_source_face_ordinal_to_authoritative_topology() {
    let imported = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();

    assert_eq!(
        imported.source_face_ids.len(),
        imported.topology.faces.len()
    );
    for (ordinal, face_id) in &imported.source_face_ids {
        assert!((*ordinal as usize) < imported.topology.faces.len());
        assert!(imported
            .topology
            .faces
            .iter()
            .any(|face| face.id == *face_id));
    }
}

#[test]
fn step_assembly_preserves_shared_definitions_occurrences_and_body_evaluation() {
    let imported = import_exact_cad(
        "two_box_assembly.step",
        TWO_BOX_ASSEMBLY,
        GeometryFormat::Step,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let reimported = import_exact_cad(
        "renamed-assembly.step",
        TWO_BOX_ASSEMBLY,
        GeometryFormat::Step,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert_eq!(imported, reimported);

    assert_eq!(imported.topology.assemblies.len(), 4);
    assert_eq!(imported.topology.instances.len(), 3);
    assert_eq!(imported.topology.bodies.len(), 2);
    assert_eq!(imported.topology.solids.len(), 2);
    assert_eq!(imported.topology.regions.len(), 2);
    assert_eq!(imported.topology.faces.len(), 12);
    assert_eq!(
        imported.source_face_ids.len(),
        imported.topology.faces.len()
    );
    assert_eq!(
        imported.source_face_ids.keys().copied().collect::<Vec<_>>(),
        (0..imported.topology.faces.len() as u64).collect::<Vec<_>>()
    );
    let body_assemblies = imported
        .topology
        .assemblies
        .iter()
        .filter(|assembly| !assembly.body_ids.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(body_assemblies.len(), 2);
    assert_eq!(
        body_assemblies[0].definition_digest,
        body_assemblies[1].definition_digest
    );
    assert_ne!(body_assemblies[0].id, body_assemblies[1].id);

    let translated = imported
        .topology
        .instances
        .iter()
        .find(|instance| instance.transform.0[3] == 10.0)
        .expect("one part occurrence must retain its translation");
    assert_eq!(translated.transform.0[7], 0.0);
    assert_eq!(translated.transform.0[11], 0.0);
    assert_eq!(
        imported
            .topology
            .vertices
            .iter()
            .filter(|vertex| vertex.id.assembly_path == translated.id.assembly_path)
            .map(|vertex| vertex.point_m[0])
            .max_by(f64::total_cmp),
        Some(1.0),
        "definition-local geometry must not absorb occurrence placement",
    );

    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    let centroids = imported
        .topology
        .bodies
        .iter()
        .map(|body| {
            runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
                &evaluator,
                &body.mass_properties_evaluator_id,
                &Unlimited,
            )
            .unwrap()
            .centroid_m
        })
        .collect::<Vec<_>>();
    assert_eq!(centroids, vec![[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]);

    let definition_limited = ExactCadImportOptions {
        max_representation_bytes: imported.representation.len() as u64,
        ..ExactCadImportOptions::default()
    };
    let definition_error = import_exact_cad(
        "two_box_assembly.step",
        TWO_BOX_ASSEMBLY,
        GeometryFormat::Step,
        &definition_limited,
        &GeometryImportContext::new(),
    )
    .unwrap_err();
    assert!(
        matches!(
            definition_error,
            GeometryImportError::ExactRepresentationCapacityExceeded { .. }
        ),
        "unexpected definition budget error: {definition_error:?}",
    );

    let occurrence_limited = ExactCadImportOptions {
        max_entities: 50,
        ..ExactCadImportOptions::default()
    };
    assert!(matches!(
        import_exact_cad(
            "two_box_assembly.step",
            TWO_BOX_ASSEMBLY,
            GeometryFormat::Step,
            &occurrence_limited,
            &GeometryImportContext::new(),
        ),
        Err(GeometryImportError::ExactEntityCapacityExceeded { limit: 50 })
    ));
}

#[test]
fn mixed_exact_definition_projects_distinct_solid_and_sheet_bodies() {
    let imported = import_exact_cad(
        "mixed_solid_sheet.brep",
        MIXED_SOLID_SHEET,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert_eq!(imported.topology.assemblies.len(), 1);
    assert_eq!(imported.topology.bodies.len(), 2);
    assert_eq!(imported.topology.lumps.len(), 1);
    assert_eq!(imported.topology.solids.len(), 1);
    assert_eq!(imported.topology.shells.len(), 2);
    assert_eq!(imported.topology.faces.len(), 7);
    assert_eq!(imported.topology.edges.len(), 16);
    assert_eq!(imported.topology.vertices.len(), 12);
    assert_eq!(imported.topology.assemblies[0].body_ids.len(), 2);

    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    let solid = imported
        .topology
        .bodies
        .iter()
        .find(|body| !body.is_sheet_body)
        .unwrap();
    let sheet = imported
        .topology
        .bodies
        .iter()
        .find(|body| body.is_sheet_body)
        .unwrap();
    assert_eq!(solid.id.source_topology_id, "body:solid");
    assert_eq!(sheet.id.source_topology_id, "body:sheet");
    assert_eq!(solid.lump_ids.len(), 1);
    assert!(solid.sheet_shell_ids.is_empty());
    assert!(sheet.lump_ids.is_empty());
    assert_eq!(sheet.sheet_shell_ids.len(), 1);

    let solid_mass = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
        &evaluator,
        &solid.mass_properties_evaluator_id,
        &Unlimited,
    )
    .unwrap();
    assert_eq!(solid_mass.volume_m3, 6.0);
    assert_eq!(solid_mass.surface_area_m2, 22.0);
    assert_eq!(solid_mass.centroid_m, [0.5, 1.0, 1.5]);

    let sheet_mass = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
        &evaluator,
        &sheet.mass_properties_evaluator_id,
        &Unlimited,
    )
    .unwrap();
    assert_eq!(sheet_mass.volume_m3, 0.0);
    assert_eq!(sheet_mass.surface_area_m2, 8.0);
    assert_eq!(sheet_mass.centroid_m, [1.0, 2.0, 10.0]);

    let mut mislabeled = imported.clone();
    let sheet_record = mislabeled
        .evaluators
        .mass_properties
        .iter_mut()
        .find(|record| record.id == sheet.mass_properties_evaluator_id)
        .unwrap();
    let runmat_geometry_core::ExactMassPropertiesImplementation::Kernel { reference } =
        &mut sheet_record.implementation
    else {
        panic!("mixed bodies require body-specific kernel evaluators");
    };
    reference.entity_token = "body:solid".into();
    assert!(matches!(
        OcctExactEvaluator::new(&mislabeled),
        Err(GeometryEvaluationError {
            kind: GeometryEvaluationErrorKind::InconsistentGeometry,
            ..
        })
    ));
}

#[test]
fn conformal_solids_share_one_oriented_interface_identity() {
    let imported = import_exact_cad(
        "shared_interface.brep",
        SHARED_INTERFACE,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let reimported = import_exact_cad(
        "renamed-interface.brep",
        SHARED_INTERFACE,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();

    assert_eq!(imported, reimported);
    assert_eq!(imported.topology.solids.len(), 2);
    assert_eq!(imported.topology.faces.len(), 11);
    assert_eq!(imported.topology.interfaces.len(), 1);
    let interface = &imported.topology.interfaces[0];
    assert_eq!(
        interface.side_a_region_id.kind,
        runmat_geometry_core::PersistentEntityKind::Region
    );
    assert_eq!(
        interface.side_b_region_id.kind,
        runmat_geometry_core::PersistentEntityKind::Region
    );
    assert_ne!(interface.side_a_region_id, interface.side_b_region_id);
    assert!(imported
        .topology
        .regions
        .iter()
        .any(|region| region.id == interface.side_a_region_id));
    assert!(imported
        .topology
        .regions
        .iter()
        .any(|region| region.id == interface.side_b_region_id));
    assert_ne!(interface.side_a_orientation, interface.side_b_orientation);
    assert_eq!(imported.model.interface_count, 1);
    assert_eq!(imported.model.region_count, 2);
    assert_eq!(imported.model.contact_count, 0);
}

#[test]
fn independent_coincident_faces_require_explicit_contact_authoring() {
    let unpaired = import_exact_cad(
        "independent_contact.step",
        INDEPENDENT_CONTACT,
        GeometryFormat::Step,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert_eq!(unpaired.topology.solids.len(), 2);
    assert_eq!(unpaired.topology.faces.len(), 12);
    assert!(unpaired.topology.interfaces.is_empty());
    assert!(unpaired.topology.contacts.is_empty());

    let evaluator = OcctExactEvaluator::new(&unpaired).unwrap();
    let mut coincident_faces = unpaired
        .topology
        .faces
        .iter()
        .filter_map(|face| {
            let bounds = runmat_geometry_core::ExactSurfaceEvaluator::parameter_bounds(
                &evaluator,
                &face.surface_evaluator_id,
            )
            .unwrap();
            let uv = [
                (bounds[0].start + bounds[0].end) * 0.5,
                (bounds[1].start + bounds[1].end) * 0.5,
            ];
            let point = runmat_geometry_core::ExactSurfaceEvaluator::point(
                &evaluator,
                &face.surface_evaluator_id,
                uv,
                &Unlimited,
            )
            .unwrap();
            let translation_x = unpaired
                .topology
                .instances
                .iter()
                .find(|instance| instance.id.assembly_path == face.id.assembly_path)
                .map(|instance| instance.transform.0[3])?;
            ((point[0] + translation_x - 1.0).abs() < 1.0e-12).then(|| face.id.clone())
        })
        .collect::<Vec<_>>();
    coincident_faces.sort();
    assert_eq!(coincident_faces.len(), 2);

    let definitions = [ExactContactDefinition {
        side_a_face_ids: vec![coincident_faces[1].clone()],
        side_b_face_ids: vec![coincident_faces[0].clone()],
    }];
    let paired = import_exact_cad(
        "renamed-contact.step",
        INDEPENDENT_CONTACT,
        GeometryFormat::Step,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap()
    .with_contacts(&definitions)
    .unwrap();
    assert!(paired.topology.interfaces.is_empty());
    assert_eq!(paired.topology.contacts.len(), 1);
    assert_eq!(paired.topology.contacts[0].side_a_face_ids.len(), 1);
    assert_eq!(paired.topology.contacts[0].side_b_face_ids.len(), 1);
    assert_eq!(paired.model.contact_count, 1);
    paired.build_closure().unwrap();
}

#[test]
fn disconnected_solids_share_one_body_with_aggregate_mass_properties() {
    let imported = import_exact_cad(
        "disconnected_solids.brep",
        DISCONNECTED_SOLIDS,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert_eq!(imported.topology.bodies.len(), 1);
    assert_eq!(imported.topology.lumps.len(), 2);
    assert_eq!(imported.topology.solids.len(), 2);
    let body = &imported.topology.bodies[0];
    assert!(!body.is_sheet_body);
    assert_eq!(body.lump_ids.len(), 2);

    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    let mass = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
        &evaluator,
        &body.mass_properties_evaluator_id,
        &Unlimited,
    )
    .unwrap();
    assert_eq!(mass.volume_m3, 12.0);
    assert_eq!(mass.surface_area_m2, 44.0);
    assert_eq!(mass.centroid_m, [5.5, 1.0, 1.5]);
}

#[test]
fn persistent_entity_names_ignore_compound_child_order_and_are_bounded() {
    let options = ExactCadImportOptions::default();
    let context = GeometryImportContext::new();
    let forward = import_exact_cad(
        "disconnected_solids.brep",
        DISCONNECTED_SOLIDS,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    let reversed = import_exact_cad(
        "disconnected_solids_reversed.brep",
        DISCONNECTED_SOLIDS_REVERSED,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    assert_ne!(forward.representation, reversed.representation);
    assert_eq!(persistent_names(&forward), persistent_names(&reversed));
    assert_eq!(evaluator_names(&forward), evaluator_names(&reversed));
    assert_ne!(curve_bindings(&forward), curve_bindings(&reversed));
    assert!(persistent_names(&forward)
        .iter()
        .all(|name| name.starts_with("occt:") && name.len() == 69));
    for imported in [&forward, &reversed] {
        let evaluator = OcctExactEvaluator::new(imported).unwrap();
        let body = &imported.topology.bodies[0];
        let mass = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
            &evaluator,
            &body.mass_properties_evaluator_id,
            &Unlimited,
        )
        .unwrap();
        assert_eq!(mass.volume_m3, 12.0);
        assert_eq!(mass.centroid_m, [5.5, 1.0, 1.5]);
    }

    let limited = ExactCadImportOptions {
        max_identity_work_bytes: 1,
        ..options
    };
    assert!(matches!(
        import_exact_cad(
            "disconnected_solids.brep",
            DISCONNECTED_SOLIDS,
            GeometryFormat::Brep,
            &limited,
            &context,
        ),
        Err(GeometryImportError::ExactValidationBudgetExceeded(reason))
            if reason.contains("persistent identity serialization")
    ));
}

#[test]
fn persistent_entity_names_reject_ambiguous_coincident_topology() {
    let error = import_exact_cad(
        "coincident_solids.brep",
        COINCIDENT_SOLIDS,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        GeometryImportError::InvalidGeometry(reason)
            if reason.contains("ambiguous coincident persistent names")
    ));
}

#[test]
fn duplicate_consolidation_collapses_indistinguishable_compound_children() {
    let mut options = ExactCadImportOptions::default();
    options.analysis.healing.consolidate_duplicates = true;
    let imported = import_exact_cad(
        "coincident_solids.brep",
        COINCIDENT_SOLIDS,
        GeometryFormat::Brep,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    let renamed = import_exact_cad(
        "renamed-coincident-solids.brep",
        COINCIDENT_SOLIDS,
        GeometryFormat::Brep,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert_eq!(imported, renamed);
    assert_eq!(imported.topology.bodies.len(), 1);
    assert_eq!(imported.topology.lumps.len(), 1);
    assert_eq!(imported.topology.solids.len(), 1);
    let report = imported.healing_report.as_ref().unwrap();
    assert_eq!(report.operations.len(), 1);
    assert_eq!(
        report.operations[0].kind,
        GeometryHealingOperationKind::ConsolidateDuplicate
    );
    assert_eq!(report.operations[0].maximum_displacement_m, 0.0);
    assert!(report.original_validity.is_valid());
    assert!(report.healed_validity.is_valid());
    assert_eq!(imported.analysis_options().revision.revision, 2);
    assert!(imported.build_closure().unwrap().healing_bytes.is_some());

    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    let mass = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
        &evaluator,
        &imported.topology.bodies[0].mass_properties_evaluator_id,
        &Unlimited,
    )
    .unwrap();
    assert_eq!(mass.volume_m3, 6.0);
    assert_eq!(mass.centroid_m, [0.5, 1.0, 1.5]);

    let unchanged = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert!(unchanged.healing_report.is_none());

    let unchanged_assembly = import_exact_cad(
        "two_box_assembly.step",
        TWO_BOX_ASSEMBLY,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert_eq!(unchanged_assembly.topology.instances.len(), 3);
    assert!(unchanged_assembly.healing_report.is_none());

    let limited = ExactCadImportOptions {
        max_identity_work_bytes: 1,
        ..options
    };
    assert!(matches!(
        import_exact_cad(
            "coincident_solids.brep",
            COINCIDENT_SOLIDS,
            GeometryFormat::Brep,
            &limited,
            &GeometryImportContext::new(),
        ),
        Err(GeometryImportError::ExactValidationBudgetExceeded(reason))
            if reason.contains("persistent identity serialization")
    ));
}

#[test]
fn gap_repair_sews_sheet_boundaries_with_measured_lineage() {
    let context = GeometryImportContext::new();
    assert!(import_exact_cad(
        "gapped_sheet.brep",
        GAPPED_SHEET,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &context,
    )
    .is_err());

    let mut options = ExactCadImportOptions::default();
    options.analysis.healing.repair_tolerance_scale_gaps = true;
    options.analysis.maximum_healing_displacement_m = 1.0e-6;
    let imported = import_exact_cad(
        "gapped_sheet.brep",
        GAPPED_SHEET,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(imported.topology.bodies.len(), 1);
    assert!(imported.topology.bodies[0].is_sheet_body);
    assert_eq!(imported.topology.faces.len(), 2);
    let report = imported.healing_report.as_ref().unwrap();
    assert_eq!(report.operations.len(), 1);
    assert_eq!(
        report.operations[0].kind,
        GeometryHealingOperationKind::RepairGap
    );
    assert!(report.operations[0].maximum_displacement_m > 0.0);
    assert!(report.operations[0].maximum_displacement_m <= 1.0e-6);
    assert!(report
        .revision_map
        .operations
        .iter()
        .any(|operation| matches!(
            operation,
            runmat_geometry_core::GeometryRevisionOperation::Merge { .. }
        )));
    assert!(imported.build_closure().unwrap().healing_bytes.is_some());

    let mut sewing_options = ExactCadImportOptions::default();
    sewing_options.analysis.healing.sew = true;
    sewing_options.analysis.maximum_healing_displacement_m = 1.0e-6;
    let sewn = import_exact_cad(
        "gapped_sheet.brep",
        GAPPED_SHEET,
        GeometryFormat::Brep,
        &sewing_options,
        &context,
    )
    .unwrap();
    assert_eq!(sewn.topology, imported.topology);
    assert_eq!(
        sewn.healing_report.as_ref().unwrap().operations[0].kind,
        GeometryHealingOperationKind::Sew
    );

    let mut too_small = options;
    too_small.analysis.maximum_healing_displacement_m = 1.0e-8;
    assert!(matches!(
        import_exact_cad(
            "gapped_sheet.brep",
            GAPPED_SHEET,
            GeometryFormat::Brep,
            &too_small,
            &context,
        ),
        Err(GeometryImportError::InvalidGeometry(_))
    ));
}

#[test]
fn small_topology_repair_simplifies_short_edges_and_sliver_faces() {
    let context = GeometryImportContext::new();
    let original_short_edge = import_exact_cad(
        "short_edge_face.brep",
        SHORT_EDGE_FACE,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &context,
    )
    .unwrap();
    assert_eq!(original_short_edge.topology.edges.len(), 5);

    let mut options = ExactCadImportOptions::default();
    options
        .analysis
        .healing
        .simplify_short_edges_and_sliver_faces = true;
    options.analysis.maximum_healing_displacement_m = 1.0e-6;
    let repaired_short_edge = import_exact_cad(
        "short_edge_face.brep",
        SHORT_EDGE_FACE,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(repaired_short_edge.topology.edges.len(), 4);
    let short_edge_report = repaired_short_edge.healing_report.as_ref().unwrap();
    assert_eq!(short_edge_report.operations.len(), 1);
    assert_eq!(
        short_edge_report.operations[0].kind,
        GeometryHealingOperationKind::SimplifyShortEdge
    );
    assert!(short_edge_report.operations[0].maximum_displacement_m > 0.0);
    assert!(short_edge_report.operations[0].maximum_displacement_m <= 1.0e-6);
    assert!(short_edge_report
        .revision_map
        .operations
        .iter()
        .any(|operation| matches!(
            operation,
            runmat_geometry_core::GeometryRevisionOperation::Delete { .. }
                | runmat_geometry_core::GeometryRevisionOperation::Replace { .. }
        )));
    assert!(repaired_short_edge
        .build_closure()
        .unwrap()
        .healing_bytes
        .is_some());
    let repeated_short_edge = import_exact_cad(
        "short_edge_face.brep",
        SHORT_EDGE_FACE,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(repeated_short_edge.topology, repaired_short_edge.topology);
    assert_eq!(
        repeated_short_edge.healing_report,
        repaired_short_edge.healing_report
    );

    let original_sliver = import_exact_cad(
        "sliver_face_sheet.brep",
        SLIVER_FACE_SHEET,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &context,
    )
    .unwrap();
    assert_eq!(original_sliver.topology.faces.len(), 3);
    let repaired_sliver = import_exact_cad(
        "sliver_face_sheet.brep",
        SLIVER_FACE_SHEET,
        GeometryFormat::Brep,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(repaired_sliver.topology.faces.len(), 2);
    let sliver_report = repaired_sliver.healing_report.as_ref().unwrap();
    assert!(sliver_report
        .operations
        .iter()
        .any(|operation| { operation.kind == GeometryHealingOperationKind::SimplifySliverFace }));
    assert!(sliver_report
        .operations
        .iter()
        .any(|operation| operation.maximum_displacement_m > 0.0));
    assert!(sliver_report
        .operations
        .iter()
        .all(|operation| operation.maximum_displacement_m <= 1.0e-6));
    assert!(sliver_report
        .revision_map
        .operations
        .iter()
        .any(|operation| matches!(
            operation,
            runmat_geometry_core::GeometryRevisionOperation::Delete { .. }
        )));
    assert!(repaired_sliver
        .build_closure()
        .unwrap()
        .healing_bytes
        .is_some());

    let mut too_small = options.clone();
    too_small.analysis.maximum_healing_displacement_m = 1.0e-8;
    let unchanged = import_exact_cad(
        "short_edge_face.brep",
        SHORT_EDGE_FACE,
        GeometryFormat::Brep,
        &too_small,
        &context,
    )
    .unwrap();
    assert_eq!(unchanged.topology.edges.len(), 5);
    assert!(unchanged.healing_report.is_none());

    let unchanged_assembly = import_exact_cad(
        "two_box_assembly.step",
        TWO_BOX_ASSEMBLY,
        GeometryFormat::Step,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(unchanged_assembly.topology.instances.len(), 3);
    assert!(unchanged_assembly.healing_report.is_none());

    let original_xcaf = import_exact_cad(
        "xcaf_short_edge_assembly.step",
        XCAF_SHORT_EDGE_ASSEMBLY,
        GeometryFormat::Step,
        &ExactCadImportOptions::default(),
        &context,
    )
    .unwrap();
    assert_eq!(original_xcaf.topology.instances.len(), 3);
    assert_eq!(original_xcaf.topology.edges.len(), 10);
    let repaired_xcaf = import_exact_cad(
        "xcaf_short_edge_assembly.step",
        XCAF_SHORT_EDGE_ASSEMBLY,
        GeometryFormat::Step,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(repaired_xcaf.topology.instances.len(), 3);
    assert_eq!(repaired_xcaf.topology.edges.len(), 8);
    let xcaf_report = repaired_xcaf.healing_report.as_ref().unwrap();
    assert_eq!(
        xcaf_report.operations[0].kind,
        GeometryHealingOperationKind::SimplifyShortEdge
    );
    let affected_paths = xcaf_report.operations[0]
        .affected_before
        .iter()
        .map(|entity| entity.assembly_path.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(affected_paths.len(), 2);
    assert!(affected_paths.iter().all(|path| path[0] == "root"));
    assert!(repaired_xcaf
        .build_closure()
        .unwrap()
        .healing_bytes
        .is_some());

    let repaired_labeled_face = import_exact_cad(
        "xcaf_labeled_face_assembly.step",
        XCAF_LABELED_FACE_ASSEMBLY,
        GeometryFormat::Step,
        &options,
        &context,
    )
    .unwrap();
    assert_eq!(repaired_labeled_face.topology.instances.len(), 3);
    assert_eq!(repaired_labeled_face.topology.edges.len(), 8);

    let deleted_label = import_exact_cad(
        "xcaf_labeled_short_edge_assembly.step",
        XCAF_LABELED_SHORT_EDGE_ASSEMBLY,
        GeometryFormat::Step,
        &options,
        &context,
    )
    .unwrap_err();
    let GeometryImportError::RevisionConflict { conflict } = deleted_label else {
        panic!("expected a typed revision conflict");
    };
    assert_eq!(conflict.kind, LabeledSubshapeRemapConflictKind::Deleted);
    assert_eq!(conflict.label_entries.len(), 1);
    assert_eq!(conflict.source_topology_ids.len(), 1);
    assert!(conflict.candidate_topology_ids.is_empty());
}

fn persistent_names(imported: &crate::ImportedExactCad) -> Vec<String> {
    let mut names = imported
        .topology
        .lumps
        .iter()
        .map(|entity| entity.id.source_topology_id.clone())
        .chain(
            imported
                .topology
                .solids
                .iter()
                .map(|entity| entity.id.source_topology_id.clone()),
        )
        .chain(
            imported
                .topology
                .shells
                .iter()
                .map(|entity| entity.id.source_topology_id.clone()),
        )
        .chain(
            imported
                .topology
                .faces
                .iter()
                .map(|entity| entity.id.source_topology_id.clone()),
        )
        .chain(
            imported
                .topology
                .wires
                .iter()
                .map(|entity| entity.id.source_topology_id.clone()),
        )
        .chain(
            imported
                .topology
                .coedges
                .iter()
                .map(|entity| entity.id.source_topology_id.clone()),
        )
        .chain(
            imported
                .topology
                .edges
                .iter()
                .map(|entity| entity.id.source_topology_id.clone()),
        )
        .chain(
            imported
                .topology
                .vertices
                .iter()
                .map(|entity| entity.id.source_topology_id.clone()),
        )
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn evaluator_names(imported: &crate::ImportedExactCad) -> Vec<String> {
    let mut names = imported
        .evaluators
        .curves
        .iter()
        .map(|record| record.id.as_str().to_owned())
        .chain(
            imported
                .evaluators
                .pcurves
                .iter()
                .map(|record| record.id.as_str().to_owned()),
        )
        .chain(
            imported
                .evaluators
                .surfaces
                .iter()
                .map(|record| record.id.as_str().to_owned()),
        )
        .chain(
            imported
                .evaluators
                .trim_classifiers
                .iter()
                .map(|record| record.id.as_str().to_owned()),
        )
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn curve_bindings(imported: &crate::ImportedExactCad) -> Vec<(String, String)> {
    let mut bindings = imported
        .evaluators
        .curves
        .iter()
        .map(|record| {
            let runmat_geometry_core::ExactCurveImplementation::Kernel { reference } =
                &record.implementation
            else {
                panic!("an OCCT import cannot contain a portable curve");
            };
            (
                record.id.as_str().to_owned(),
                reference.entity_token.clone(),
            )
        })
        .collect::<Vec<_>>();
    bindings.sort();
    bindings
}

#[test]
fn cavity_import_preserves_outer_and_void_shell_nesting() {
    let imported = import_exact_cad(
        "box_cavity.brep",
        BOX_CAVITY,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    assert_eq!(imported.topology.bodies.len(), 1);
    assert_eq!(imported.topology.lumps.len(), 1);
    assert_eq!(imported.topology.solids.len(), 1);
    assert_eq!(imported.topology.shells.len(), 2);
    assert_eq!(imported.topology.faces.len(), 12);
    assert_eq!(imported.topology.edges.len(), 24);
    assert_eq!(imported.topology.vertices.len(), 16);
    assert_eq!(imported.topology.solids[0].void_shell_ids.len(), 1);

    let body = &imported.topology.bodies[0];
    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    let mass = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
        &evaluator,
        &body.mass_properties_evaluator_id,
        &Unlimited,
    )
    .unwrap();
    assert!((mass.volume_m3 - 992.0).abs() < 1.0e-10);
    assert!((mass.surface_area_m2 - 624.0).abs() < 1.0e-10);
    assert!(mass
        .centroid_m
        .iter()
        .all(|coordinate| (coordinate - 5.0).abs() < 1.0e-12));
}

#[test]
fn cavity_import_rejects_a_void_shell_outside_its_outer_shell() {
    let error = import_exact_cad(
        "invalid_cavity.brep",
        INVALID_CAVITY,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        GeometryImportError::InvalidGeometry(reason)
            if reason.contains("void shell is not nested inside its outer shell")
    ));
}

struct Cancelled;

impl GeometryEvaluationControl for Cancelled {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::Cancelled,
            "test cancellation",
        ))
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        unreachable!()
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        unreachable!()
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        unreachable!()
    }
}

#[test]
fn imported_curve_queries_are_exact_scaled_and_digest_bound() {
    let imported = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let raw_digest: [u8; 32] = sha2::Sha256::digest(&imported.representation).into();
    assert_eq!(imported.representation_digest(), raw_digest);
    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    evaluator
        .validate_incidence_consistency(&imported.topology, 1.0e-9, &Unlimited)
        .unwrap();
    assert_eq!(
        evaluator
            .validate_incidence_consistency(&imported.topology, 1.0e-9, &Cancelled)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    let mut inconsistent_topology = imported.topology.clone();
    inconsistent_topology.vertices[0].point_m[0] += 1.0;
    assert_eq!(
        evaluator
            .validate_incidence_consistency(&inconsistent_topology, 1.0e-9, &Unlimited)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
    let mass_id = &imported.topology.bodies[0].mass_properties_evaluator_id;
    let mass_properties = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
        &evaluator, mass_id, &Unlimited,
    )
    .unwrap();
    assert!((mass_properties.volume_m3 - 6.0).abs() < 1.0e-12);
    assert!((mass_properties.surface_area_m2 - 22.0).abs() < 1.0e-12);
    assert_eq!(
        runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
            &evaluator, mass_id, &Cancelled,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    assert_eq!(
        runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
            &evaluator,
            &MassPropertiesEvaluatorId::new("mass:unknown").unwrap(),
            &Unlimited,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::UnknownEvaluator
    );
    let id = &imported.topology.edges[0].curve_evaluator_id;
    let range = evaluator.parameter_range(id).unwrap();
    let start = evaluator.point(id, range.start, &Unlimited).unwrap();
    let end = evaluator.point(id, range.end, &Unlimited).unwrap();
    let expected_length = norm([end[0] - start[0], end[1] - start[1], end[2] - start[2]]);
    let length = evaluator
        .arc_length_m(id, range, 1.0e-12, &Unlimited)
        .unwrap();
    assert!((length - expected_length).abs() < 1.0e-12);

    let parameter = (range.start + range.end) * 0.5;
    let point = evaluator.point(id, parameter, &Unlimited).unwrap();
    let tangent = evaluator.unit_tangent(id, parameter, &Unlimited).unwrap();
    assert!((norm(tangent) - 1.0).abs() < 1.0e-12);
    assert_eq!(
        evaluator
            .curvature_1_per_m(id, parameter, &Unlimited)
            .unwrap(),
        0.0
    );
    let projection = evaluator
        .inverse_project(id, point, 1.0e-12, &Unlimited)
        .unwrap();
    assert!((projection.parameter - parameter).abs() < 1.0e-12);
    assert!(projection.distance_m < 1.0e-12);
    assert_eq!(
        evaluator
            .point(id, range.end + 1.0, &Unlimited)
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
    assert_eq!(
        evaluator.point(id, parameter, &Cancelled).unwrap_err().kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    assert_eq!(
        evaluator
            .parameter_range(&CurveEvaluatorId::new("curve:unknown").unwrap())
            .unwrap_err()
            .kind,
        GeometryEvaluationErrorKind::UnknownEvaluator
    );

    let pcurve_id = &imported.topology.coedges[0].pcurve_evaluator_id;
    let pcurve_range =
        runmat_geometry_core::ExactPcurveEvaluator::parameter_range(&evaluator, pcurve_id).unwrap();
    let pcurve_parameter = (pcurve_range.start + pcurve_range.end) * 0.5;
    let pcurve = runmat_geometry_core::ExactPcurveEvaluator::derivatives(
        &evaluator,
        pcurve_id,
        pcurve_parameter,
        &Unlimited,
    )
    .unwrap();
    assert!(pcurve
        .point_uv
        .into_iter()
        .chain(pcurve.first_uv)
        .chain(pcurve.second_uv)
        .all(f64::is_finite));
    for coedge in &imported.topology.coedges {
        let range = runmat_geometry_core::ExactPcurveEvaluator::parameter_range(
            &evaluator,
            &coedge.pcurve_evaluator_id,
        )
        .unwrap();
        runmat_geometry_core::ExactPcurveEvaluator::derivatives(
            &evaluator,
            &coedge.pcurve_evaluator_id,
            (range.start + range.end) * 0.5,
            &Unlimited,
        )
        .unwrap();
    }
    assert_eq!(
        runmat_geometry_core::ExactPcurveEvaluator::point(
            &evaluator,
            pcurve_id,
            pcurve_range.end + 1.0,
            &Unlimited,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
    assert_eq!(
        runmat_geometry_core::ExactPcurveEvaluator::point(
            &evaluator,
            pcurve_id,
            pcurve_parameter,
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::Cancelled
    );

    let coedge = &imported.topology.coedges[0];
    let face = imported
        .topology
        .faces
        .iter()
        .find(|face| face.id == coedge.face_id)
        .unwrap();
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &face.trim_classifier_id,
            pcurve.point_uv,
            1.0e-9,
            &Unlimited,
        )
        .unwrap(),
        TrimDomainLocation::OnBoundary
    );
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &face.trim_classifier_id,
            [pcurve.point_uv[0] + 1.0e6, pcurve.point_uv[1] + 1.0e6],
            1.0e-9,
            &Unlimited,
        )
        .unwrap(),
        TrimDomainLocation::Outside
    );
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &face.trim_classifier_id,
            pcurve.point_uv,
            1.0e-9,
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    assert_eq!(
        runmat_geometry_core::ExactTrimClassifier::classify(
            &evaluator,
            &TrimClassifierId::new("trim:unknown").unwrap(),
            pcurve.point_uv,
            1.0e-9,
            &Unlimited,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::UnknownEvaluator
    );

    let surface_id = &face.surface_evaluator_id;
    let surface_bounds =
        runmat_geometry_core::ExactSurfaceEvaluator::parameter_bounds(&evaluator, surface_id)
            .unwrap();
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::periodicity(&evaluator, surface_id).unwrap(),
        [None, None]
    );
    let surface_uv = [
        (surface_bounds[0].start + surface_bounds[0].end) * 0.5,
        (surface_bounds[1].start + surface_bounds[1].end) * 0.5,
    ];
    let surface_derivatives = runmat_geometry_core::ExactSurfaceEvaluator::derivatives(
        &evaluator, surface_id, surface_uv, &Unlimited,
    )
    .unwrap();
    let normal = runmat_geometry_core::ExactSurfaceEvaluator::unit_normal(
        &evaluator, surface_id, surface_uv, &Unlimited,
    )
    .unwrap();
    assert!((norm(normal) - 1.0).abs() < 1.0e-12);
    let curvature = runmat_geometry_core::ExactSurfaceEvaluator::principal_curvature(
        &evaluator, surface_id, surface_uv, &Unlimited,
    )
    .unwrap();
    assert!(curvature.minimum_1_per_m.abs() < 1.0e-12);
    assert!(curvature.maximum_1_per_m.abs() < 1.0e-12);
    let displaced =
        std::array::from_fn(|axis| surface_derivatives.point_m[axis] + normal[axis] * 0.25);
    let surface_projection = runmat_geometry_core::ExactSurfaceEvaluator::closest_point(
        &evaluator, surface_id, displaced, 1.0e-12, &Unlimited,
    )
    .unwrap();
    assert!((surface_projection.distance_m - 0.25).abs() < 1.0e-12);
    assert!(
        norm(std::array::from_fn(|axis| {
            surface_projection.point_m[axis] - surface_derivatives.point_m[axis]
        })) < 1.0e-12
    );
    let u_boundary_uv = [surface_bounds[0].end, surface_uv[1]];
    let u_boundary = runmat_geometry_core::ExactSurfaceEvaluator::point(
        &evaluator,
        surface_id,
        u_boundary_uv,
        &Unlimited,
    )
    .unwrap();
    let u_direction = normalized(surface_derivatives.du_m).unwrap();
    let beyond_u = std::array::from_fn(|axis| u_boundary[axis] + u_direction[axis] * 0.25);
    let boundary_projection = runmat_geometry_core::ExactSurfaceEvaluator::closest_point(
        &evaluator, surface_id, beyond_u, 1.0e-12, &Unlimited,
    )
    .unwrap();
    assert!((boundary_projection.uv[0] - surface_bounds[0].end).abs() < 1.0e-12);
    assert!((boundary_projection.distance_m - 0.25).abs() < 1.0e-12);

    let edge = imported
        .topology
        .edges
        .iter()
        .find(|edge| edge.id == coedge.edge_id)
        .unwrap();
    let boundary_3d = runmat_geometry_core::ExactSurfaceEvaluator::point(
        &evaluator,
        surface_id,
        pcurve.point_uv,
        &Unlimited,
    )
    .unwrap();
    let edge_3d = evaluator
        .point(&edge.curve_evaluator_id, pcurve_parameter, &Unlimited)
        .unwrap();
    assert!(norm(std::array::from_fn(|axis| boundary_3d[axis] - edge_3d[axis])) < 1.0e-12);
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::point(
            &evaluator,
            surface_id,
            [surface_bounds[0].end + 1.0, surface_uv[1]],
            &Unlimited,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    );
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::point(
            &evaluator, surface_id, surface_uv, &Cancelled,
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::Cancelled
    );
    assert_eq!(
        runmat_geometry_core::ExactSurfaceEvaluator::parameter_bounds(
            &evaluator,
            &SurfaceEvaluatorId::new("surface:unknown").unwrap(),
        )
        .unwrap_err()
        .kind,
        GeometryEvaluationErrorKind::UnknownEvaluator
    );

    let millimeter_options = ExactCadImportOptions {
        source_units: UnitSystem::Millimeter,
        ..ExactCadImportOptions::default()
    };
    let millimeter_import = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &millimeter_options,
        &GeometryImportContext::new(),
    )
    .unwrap();
    let millimeter_evaluator = OcctExactEvaluator::new(&millimeter_import).unwrap();
    let millimeter_length = millimeter_evaluator
        .arc_length_m(
            &millimeter_import.topology.edges[0].curve_evaluator_id,
            range,
            1.0e-15,
            &Unlimited,
        )
        .unwrap();
    assert!((millimeter_length - length * 0.001).abs() < 1.0e-15);
    let millimeter_surface_id = &millimeter_import
        .topology
        .faces
        .iter()
        .find(|candidate| candidate.id == face.id)
        .unwrap()
        .surface_evaluator_id;
    let millimeter_surface_point = runmat_geometry_core::ExactSurfaceEvaluator::point(
        &millimeter_evaluator,
        millimeter_surface_id,
        surface_uv,
        &Unlimited,
    )
    .unwrap();
    assert!(
        norm(std::array::from_fn(|axis| {
            millimeter_surface_point[axis] - surface_derivatives.point_m[axis] * 0.001
        })) < 1.0e-15
    );
    let millimeter_displaced =
        std::array::from_fn(|axis| millimeter_surface_point[axis] + normal[axis] * 0.00025);
    let millimeter_projection = runmat_geometry_core::ExactSurfaceEvaluator::closest_point(
        &millimeter_evaluator,
        millimeter_surface_id,
        millimeter_displaced,
        1.0e-15,
        &Unlimited,
    )
    .unwrap();
    assert!((millimeter_projection.distance_m - 0.00025).abs() < 1.0e-15);
    let millimeter_mass = runmat_geometry_core::ExactMassPropertiesEvaluator::mass_properties(
        &millimeter_evaluator,
        &millimeter_import.topology.bodies[0].mass_properties_evaluator_id,
        &Unlimited,
    )
    .unwrap();
    assert!((millimeter_mass.volume_m3 - mass_properties.volume_m3 * 1.0e-9).abs() < 1.0e-20);
    assert!(
        (millimeter_mass.surface_area_m2 - mass_properties.surface_area_m2 * 1.0e-6).abs()
            < 1.0e-18
    );

    let mut corrupt = imported.clone();
    corrupt.representation[0] ^= 1;
    assert_eq!(
        OcctExactEvaluator::new(&corrupt).err().unwrap().kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
    let mut corrupt_mass = imported.clone();
    let runmat_geometry_core::ExactMassPropertiesImplementation::KernelValidated {
        properties, ..
    } = &mut corrupt_mass.evaluators.mass_properties[0].implementation
    else {
        panic!("box mass properties must be kernel validated")
    };
    properties.volume_m3 += 1.0;
    assert_eq!(
        OcctExactEvaluator::new(&corrupt_mass).err().unwrap().kind,
        GeometryEvaluationErrorKind::InconsistentGeometry
    );
}
