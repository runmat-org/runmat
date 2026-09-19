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
    assert_eq!(beam.beam_materials().len(), 1);
    assert_eq!(beam.shell_materials().len(), 0);
    assert!(beam.identity.mesh_digest.is_none());
    assert!(beam.identity.explicit_topology_digest.is_some());

    let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    let shell = compile_property_map(&shell_model, None).expect("compiled shell properties");
    assert_eq!(shell.beam_materials().len(), 0);
    assert_eq!(shell.shell_materials().len(), 1);

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
