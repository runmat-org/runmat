use crate::fixtures::{fixture_model, FixtureId};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use super::*;

#[test]
fn compiled_map_separates_semantic_identity_from_audit_confidence() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let first = compile_solid_property_map(&model, &mesh).expect("compiled properties");

    let mut changed_audit = model.clone();
    changed_audit.material_assignments[0].confidence =
        runmat_analysis_core::EvidenceConfidence::Inferred;
    changed_audit.material_assignments[0].expected_material_id = "imported_steel_alias".into();
    let second = compile_solid_property_map(&changed_audit, &mesh).expect("compiled properties");

    assert_eq!(
        first.identity.semantic_digest,
        second.identity.semantic_digest
    );
    assert_ne!(first.audit_digest, second.audit_digest);
    assert_eq!(
        first.solid_materials().len(),
        mesh.topology.volume_elements.len()
    );
}

#[test]
fn compiled_map_has_a_canonical_json_safe_round_trip() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let compiled =
        compile_solid_property_map(&matching_model(&mesh), &mesh).expect("compiled properties");
    let encoded = serde_json::to_vec(&compiled).expect("encode compiled property map");
    let decoded: CompiledPropertyMap =
        serde_json::from_slice(&encoded).expect("decode compiled property map");
    assert_eq!(decoded, compiled);
}

#[test]
fn material_display_evidence_changes_audit_but_not_numerical_semantics() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let first = compile_solid_property_map(&model, &mesh).expect("compiled properties");
    let mut changed_evidence = model;
    changed_evidence.materials[0].name = "Imported display label".into();
    let second = compile_solid_property_map(&changed_evidence, &mesh).expect("compiled properties");

    assert_eq!(
        first.identity.semantic_digest,
        second.identity.semantic_digest
    );
    assert_ne!(first.audit_digest, second.audit_digest);
}

#[test]
fn compiled_map_decode_rejects_semantic_audit_and_entry_substitution() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let compiled =
        compile_solid_property_map(&matching_model(&mesh), &mesh).expect("compiled properties");
    let encoded = serde_json::to_value(&compiled).expect("encode compiled property map");

    let mut semantic = encoded.clone();
    semantic["identity"]["semantic_digest"] =
        serde_json::to_value(runmat_execution::Digest::sha256(b"substituted semantics")).unwrap();
    assert!(serde_json::from_value::<CompiledPropertyMap>(semantic).is_err());

    let mut audit = encoded.clone();
    audit["audit_digest"] =
        serde_json::to_value(runmat_execution::Digest::sha256(b"substituted audit")).unwrap();
    assert!(serde_json::from_value::<CompiledPropertyMap>(audit).is_err());

    let mut entry = encoded.clone();
    entry["solid_materials"][0][1]["element_identity"] =
        serde_json::to_value(runmat_meshing_core::StableDigest::from_bytes([0xa5; 32])).unwrap();
    assert!(serde_json::from_value::<CompiledPropertyMap>(entry).is_err());

    let mut incomplete_audit = encoded;
    incomplete_audit["audit"] = serde_json::json!([]);
    assert!(serde_json::from_value::<CompiledPropertyMap>(incomplete_audit).is_err());

    let mut material_evidence = serde_json::to_value(&compiled).unwrap();
    material_evidence["solid_materials"][0][1]["material"]["name"] =
        serde_json::json!("substituted material display name");
    assert!(serde_json::from_value::<CompiledPropertyMap>(material_evidence).is_err());
}

#[test]
fn compiled_map_rejects_gaps_overlaps_wrong_roles_and_stale_scopes() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);

    let mut gap = model.clone();
    gap.material_assignments.clear();
    assert!(matches!(
        compile_solid_property_map(&gap, &mesh),
        Err(PropertyCompilationError::InvalidCoverage {
            assignment_count: 0,
            ..
        })
    ));

    let mut overlap = model.clone();
    let mut all = overlap.material_assignments[0].clone();
    all.target = runmat_analysis_core::PropertyRegionTarget::all_solids();
    overlap.material_assignments.push(all);
    assert!(matches!(
        compile_solid_property_map(&overlap, &mesh),
        Err(PropertyCompilationError::InvalidCoverage {
            assignment_count: 2,
            ..
        })
    ));

    let mut wrong_role = model.clone();
    wrong_role.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::all_cells();
    assert!(matches!(
        compile_solid_property_map(&wrong_role, &mesh),
        Err(PropertyCompilationError::InvalidCoverage {
            assignment_count: 0,
            ..
        })
    ));

    let mut stale = model;
    stale.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::solid("absent-region");
    assert!(compile_solid_property_map(&stale, &mesh).is_err());
}

#[test]
fn structural_solid_capabilities_fail_before_assembly_without_restricting_generic_properties() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let generic = compile_property_map(&model, Some(&mesh)).expect("generic solid properties");
    let structural =
        compile_structural_property_map(&model, Some(&mesh)).expect("structural solid properties");
    assert_eq!(generic.identity, structural.identity);

    let mut thermal_only = model.clone();
    thermal_only.materials[0].elasticity = None;
    compile_property_map(&thermal_only, Some(&mesh))
        .expect("generic material ownership must not require elasticity");
    assert!(matches!(
        compile_structural_property_map(&thermal_only, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "structural solid",
            capability: runmat_analysis_core::MaterialCapability::IsotropicLinearElasticity,
            ..
        })
    ));

    let mut massless = model;
    massless.materials[0].mass_density_kg_per_m3 = None;
    compile_property_map(&massless, Some(&mesh))
        .expect("generic material ownership must not require density");
    assert!(matches!(
        compile_structural_property_map(&massless, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "structural solid",
            capability: runmat_analysis_core::MaterialCapability::MassDensity,
            ..
        })
    ));
}

#[test]
fn thermal_solid_capabilities_fail_before_assembly_without_restricting_generic_properties() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let model = matching_model(&mesh);
    let generic = compile_property_map(&model, Some(&mesh)).expect("generic solid properties");
    let thermal =
        compile_thermal_property_map(&model, Some(&mesh)).expect("thermal solid properties");
    assert_eq!(generic.identity, thermal.identity);

    let mut nonthermal = model.clone();
    nonthermal.materials[0].thermal = None;
    compile_property_map(&nonthermal, Some(&mesh)).expect("generic material ownership");
    assert!(matches!(
        compile_thermal_property_map(&nonthermal, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "thermal solid",
            capability: runmat_analysis_core::MaterialCapability::Thermal,
            ..
        })
    ));

    let mut massless = model;
    massless.materials[0].mass_density_kg_per_m3 = None;
    assert!(matches!(
        compile_thermal_property_map(&massless, Some(&mesh)),
        Err(PropertyCompilationError::MissingCapability {
            role: "thermal solid",
            capability: runmat_analysis_core::MaterialCapability::MassDensity,
            ..
        })
    ));
}

#[test]
fn compiled_map_uses_bound_entity_sets_instead_of_logical_region_names() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let selected = runmat_geometry_core::PersistentEntityId {
        kind: runmat_geometry_core::PersistentEntityKind::Solid,
        source_topology_id: "selected-solid".into(),
        assembly_path: vec!["root".into()],
    };
    mesh.topology.volume_elements[0]
        .provenance
        .push(selected.clone());
    mesh.topology.volume_elements[0].provenance.sort();
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let mut model = matching_model(&mesh);
    let mut target = runmat_analysis_core::PropertyRegionTarget::solid("logical-body");
    target
        .bind_resolved_entities(vec![
            selected,
            runmat_geometry_core::PersistentEntityId {
                kind: runmat_geometry_core::PersistentEntityKind::Solid,
                source_topology_id: "another-solid".into(),
                assembly_path: vec!["root".into()],
            },
        ])
        .unwrap();
    model.material_assignments[0].target = target;

    let compiled = compile_solid_property_map(&model, &mesh).expect("compiled properties");
    assert_eq!(compiled.solid_materials().len(), 1);

    let mut wrong_kind = runmat_analysis_core::PropertyRegionTarget::solid("logical-body");
    assert!(wrong_kind
        .bind_resolved_entities(vec![mesh.topology.boundary_faces[0].provenance[0].clone()])
        .is_err());
}

#[test]
fn compiled_map_projects_explicit_beam_and_shell_domains_without_cross_role_fallback() {
    let beam_model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let beam = compile_property_map(&beam_model, None).expect("compiled beam properties");
    assert_eq!(beam.beam_properties().len(), 1);
    assert_eq!(beam.shell_properties().len(), 0);
    assert!(beam.identity.mesh_digest.is_none());
    assert!(beam.identity.explicit_topology_digest.is_some());

    let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    let shell = compile_property_map(&shell_model, None).expect("compiled shell properties");
    assert_eq!(shell.beam_properties().len(), 0);
    assert_eq!(shell.shell_properties().len(), 1);

    let mut wrong_role = beam_model;
    wrong_role.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::all_shells();
    assert!(matches!(
        compile_property_map(&wrong_role, None),
        Err(PropertyCompilationError::InvalidCoverage {
            role: "beam",
            assignment_count: 0,
            ..
        })
    ));
}

#[test]
fn laminate_property_identity_round_trips_and_rejects_competing_homogeneous_ownership() {
    use runmat_analysis_core::{
        LaminatePlyModel, LaminateShellSectionModel, PlyOrderDirection, ShellReferenceSurface,
        ShellSectionModel,
    };

    let mut model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    model.material_assignments.clear();
    model
        .structural
        .as_mut()
        .expect("shell fixture")
        .shell_sections = vec![ShellSectionModel::Laminate(LaminateShellSectionModel {
        section_id: "panel_2mm".into(),
        reference_orientation_id: "global_x".into(),
        ply_order: PlyOrderDirection::AlongSurfaceNormal,
        reference_surface: ShellReferenceSurface::MidSurface,
        plies: ["lower", "upper"]
            .map(|ply_id| LaminatePlyModel {
                ply_id: ply_id.into(),
                material_id: "mat_steel".into(),
                thickness_m: 0.001,
                angle_rad: 0.0,
            })
            .to_vec(),
        shear_correction: Some(5.0 / 6.0),
    })];

    let compiled = compile_property_map(&model, None).expect("compiled laminate property map");
    assert_eq!(compiled.shell_properties().len(), 1);
    assert!(matches!(
        compiled.shell_properties().next().unwrap().owner,
        CompiledShellPropertyOwner::Laminate { .. }
    ));
    assert!(matches!(
        compiled.audit()[0].owner,
        CompiledPropertyAuditOwner::Laminate { .. }
    ));
    let encoded = serde_json::to_vec(&compiled).expect("encode laminate property map");
    let decoded: CompiledPropertyMap =
        serde_json::from_slice(&encoded).expect("decode laminate property map");
    assert_eq!(decoded, compiled);

    let mut missing_elasticity = model.clone();
    missing_elasticity.materials[0].elasticity = None;
    assert!(matches!(
        compile_property_map(&missing_elasticity, None),
        Err(PropertyCompilationError::MissingCapability {
            capability: runmat_analysis_core::MaterialCapability::LaminaElasticity,
            ..
        })
    ));

    let mut overlap = model;
    overlap
        .material_assignments
        .push(runmat_analysis_core::MaterialAssignment {
            target: runmat_analysis_core::PropertyRegionTarget::all_shells(),
            expected_material_id: "mat_steel".into(),
            assigned_material_id: "mat_steel".into(),
            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
        });
    assert!(matches!(
        compile_property_map(&overlap, None),
        Err(PropertyCompilationError::LaminateMaterialOverlap {
            assignment_count: 1,
            ..
        })
    ));
}

#[test]
fn semantic_identity_is_independent_of_assignment_order_but_audit_is_not() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = matching_model(&mesh);
    let mut unused_role = model.material_assignments[0].clone();
    unused_role.target = runmat_analysis_core::PropertyRegionTarget::all_cells();
    model.material_assignments.push(unused_role);

    let first = compile_solid_property_map(&model, &mesh).expect("compiled properties");
    model.material_assignments.reverse();
    let second = compile_solid_property_map(&model, &mesh).expect("compiled reordered properties");

    assert_eq!(
        first.identity.semantic_digest,
        second.identity.semantic_digest
    );
    assert_eq!(
        first.solid_materials().collect::<Vec<_>>(),
        second.solid_materials().collect::<Vec<_>>()
    );
    assert_ne!(first.audit_digest, second.audit_digest);
}

fn matching_model(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_revision = u32::try_from(mesh.geometry.geometry_revision).unwrap();
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        mesh.topology.volume_elements[0]
            .region_id
            .source_topology_id
            .clone(),
    );
    model
}
