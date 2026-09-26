//! The independently materialized legacy contract is a byte/identity reference, not a solver
//! reference. Keep its explicit arrays so these tests detect changes in borrowed sequence views.
use super::*;
use serde_json::json;

pub(crate) fn assert_legacy_identity(compiled: &CompiledPropertyMap) {
    let solids = compiled
        .solid_materials()
        .map(|entry| {
            json!({
                "element_id": entry.element_id,
                "element_identity": entry.element_identity,
                "material": entry.material.solver_identity_contract(),
            })
        })
        .collect::<Vec<_>>();
    let beams = compiled
        .beam_properties()
        .map(|entry| {
            json!({
                "element_id": entry.element_id,
                "element_identity": entry.element_identity,
                "material": entry.material.solver_identity_contract(),
                "section": entry.section,
                "reference_axis": entry.reference_axis,
            })
        })
        .collect::<Vec<_>>();
    let shells = compiled
        .shell_properties()
        .map(|entry| {
            let owner = match &entry.owner {
                CompiledShellPropertyOwner::Homogeneous { material, section } => json!({
                    "kind": "homogeneous", "material": material.solver_identity_contract(),
                    "section": section,
                }),
                CompiledShellPropertyOwner::Laminate { section, plies } => json!({
                    "kind": "laminate", "section": section,
                    "plies": plies.iter().map(|ply| json!({
                        "definition": ply.definition,
                        "material": ply.material.solver_identity_contract(),
                    })).collect::<Vec<_>>(),
                }),
            };
            json!({"element_id": entry.element_id, "element_identity": entry.element_identity,
            "reference_axis": entry.reference_axis, "owner": owner})
        })
        .collect::<Vec<_>>();
    let semantic = json!({
        "geometry_id": compiled.identity.geometry_id,
        "geometry_revision": compiled.identity.geometry_revision,
        "mesh_digest": compiled.identity.mesh_digest,
        "explicit_topology_digest": compiled.identity.explicit_topology_digest,
        "solids": solids, "beams": beams, "shells": shells,
    });
    let source = CompiledPropertySource {
        geometry_id: compiled.identity.geometry_id.clone(),
        geometry_revision: compiled.identity.geometry_revision,
        mesh_digest: compiled.identity.mesh_digest,
        explicit_topology_digest: compiled.identity.explicit_topology_digest,
    };
    let borrowed = SemanticContract {
        geometry_id: &source.geometry_id,
        geometry_revision: source.geometry_revision,
        mesh_digest: source.mesh_digest,
        explicit_topology_digest: source.explicit_topology_digest,
        solids: compiled.solid_entries(),
        beams: compiled.beam_entries(),
        shells: compiled.shell_entries(),
    };
    let expected = encode_contract(
        SEMANTIC_PREFIX,
        "compiled-property-semantics",
        &semantic,
        LIMITS,
    )
    .unwrap();
    assert_eq!(
        expected,
        encode_contract(
            SEMANTIC_PREFIX,
            "compiled-property-semantics",
            &borrowed,
            LIMITS
        )
        .unwrap()
    );
    assert_eq!(Digest::sha256(expected), compiled.identity.semantic_digest);
    let audit = json!({
        "semantic_digest": compiled.identity.semantic_digest,
        "solids": compiled.solid_materials().collect::<Vec<_>>(),
        "beams": compiled.beam_properties().collect::<Vec<_>>(),
        "shells": compiled.shell_properties().collect::<Vec<_>>(),
        "assignments": compiled.audit(),
    });
    let borrowed = AuditContract {
        semantic_digest: compiled.identity.semantic_digest,
        solids: compiled.solid_entries(),
        beams: compiled.beam_entries(),
        shells: compiled.shell_entries(),
        assignments: compiled.audit(),
    };
    let expected =
        encode_contract(AUDIT_PREFIX, "compiled-property-audit", &audit, LIMITS).unwrap();
    assert_eq!(
        expected,
        encode_contract(AUDIT_PREFIX, "compiled-property-audit", &borrowed, LIMITS).unwrap()
    );
    assert_eq!(Digest::sha256(expected), compiled.audit_digest);
}

#[test]
fn borrowed_identity_views_preserve_solid_beam_and_shell_bytes() {
    use crate::fixtures::{fixture_model, FixtureId};
    use crate::properties::{compile_property_map, compile_solid_property_map};
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.geometry_revision = u32::try_from(mesh.geometry.geometry_revision).unwrap();
        model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
            mesh.topology.volume_elements[0]
                .region_id
                .source_topology_id
                .clone(),
        );
        assert_legacy_identity(&compile_solid_property_map(&model, &mesh).unwrap());
    }
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        assert_legacy_identity(&compile_property_map(&fixture_model(fixture), None).unwrap());
    }
}
