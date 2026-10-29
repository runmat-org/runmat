//! The independently materialized reference contract is a byte/identity reference, not a solver
//! reference. Keep its explicit arrays so these tests detect changes in borrowed sequence views.
use super::*;
use serde_json::json;

pub(crate) fn assert_reference_identity(compiled: &CompiledPropertyMap) {
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
    let cells = compiled
        .cell_entries()
        .values()
        .map(|entry| {
            json!({
                "element_id": entry.element_id,
                "element_identity": entry.element_identity,
                "material": entry.material.solver_identity_contract(),
            })
        })
        .collect::<Vec<_>>();
    let semantic_sequence = |field: &str, items: &[serde_json::Value]| -> serde_json::Value {
        independent_sequence(
            SEMANTIC_ELEMENT_PREFIX,
            &format!("compiled-property-semantics.{field}"),
            items,
        )
    };
    let mut semantic = json!({
        "geometry_id": compiled.identity.geometry_id,
        "geometry_revision": compiled.identity.geometry_revision,
        "mesh_digest": compiled.identity.mesh_digest,
        "explicit_topology_digest": compiled.identity.explicit_topology_digest,
        "solids": semantic_sequence("solids", &solids),
        "beams": semantic_sequence("beams", &beams),
        "shells": semantic_sequence("shells", &shells),
    });
    if !cells.is_empty() {
        semantic["cells"] = semantic_sequence("cells", &cells);
    }
    let expected = encode_contract(
        SEMANTIC_PREFIX,
        "compiled-property-semantics",
        &semantic,
        HEADER_LIMITS,
    )
    .unwrap();
    assert_eq!(Digest::sha256(expected), compiled.identity.semantic_digest);
    let audit_sequence = |field: &str, items: Vec<serde_json::Value>| -> serde_json::Value {
        independent_sequence(
            AUDIT_ELEMENT_PREFIX,
            &format!("compiled-property-audit.{field}"),
            &items,
        )
    };
    let mut audit = json!({
        "semantic_digest": compiled.identity.semantic_digest,
        "solids": audit_sequence("solids", compiled.solid_materials().map(|entry| json!(entry)).collect()),
        "beams": audit_sequence("beams", compiled.beam_properties().map(|entry| json!(entry)).collect()),
        "shells": audit_sequence("shells", compiled.shell_properties().map(|entry| json!(entry)).collect()),
        "assignments": audit_sequence("assignments", compiled.audit().iter().map(|entry| json!(entry)).collect()),
    });
    if !compiled.cell_entries().is_empty() {
        audit["cells"] = audit_sequence(
            "cells",
            compiled
                .cell_entries()
                .values()
                .map(|entry| json!(entry))
                .collect(),
        );
    }
    let expected = encode_contract(
        AUDIT_PREFIX,
        "compiled-property-audit",
        &audit,
        HEADER_LIMITS,
    )
    .unwrap();
    assert_eq!(Digest::sha256(expected), compiled.audit_digest);
}

/// Independent reference for a streamed sequence digest: the length-delimited canonical
/// element encodings hashed as one materialized byte string.
fn independent_sequence(
    element_prefix: &[u8],
    domain: &str,
    items: &[serde_json::Value],
) -> serde_json::Value {
    let mut stream = SEQUENCE_DIGEST_PREFIX.to_vec();
    stream.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    stream.extend_from_slice(domain.as_bytes());
    for item in items {
        let bytes = encode_contract(element_prefix, domain, item, ELEMENT_LIMITS).unwrap();
        stream.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        stream.extend_from_slice(&bytes);
    }
    json!({ "count": items.len() as u64, "digest": Digest::sha256(stream) })
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
                .analysis_selection_id()
                .unwrap(),
        );
        assert_reference_identity(&compile_solid_property_map(&model, &mesh).unwrap());
    }
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        assert_reference_identity(&compile_property_map(&fixture_model(fixture), None).unwrap());
    }
}

// The current-sensor study (1,378,142 Tet10 elements, 1.85M nodes) needed a 536,870,914-byte
// v1 identity buffer against its 512 MiB ceiling. Streamed sequence digests carry no
// element-count bound: a map of that size compiles its identity, and the header stays small.
#[test]
fn property_identity_of_a_current_sensor_sized_map_needs_no_mesh_sized_buffer() {
    use crate::fixtures::{fixture_model, FixtureId};
    use crate::properties::compile_solid_property_map;
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_revision = u32::try_from(mesh.geometry.geometry_revision).unwrap();
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        mesh.topology.volume_elements[0]
            .region_id
            .analysis_selection_id()
            .unwrap(),
    );
    let compiled = compile_solid_property_map(&model, &mesh).unwrap();
    let template = compiled.solid_entries().values().next().unwrap().clone();
    const ELEMENTS: u64 = 1_378_142;
    let solids = (0..ELEMENTS)
        .map(|index| {
            let mut identity = [0u8; 32];
            identity[..8].copy_from_slice(&index.to_be_bytes());
            let identity = StableDigest::from_bytes(identity);
            let mut entry = template.clone();
            entry.element_id = index;
            entry.element_identity = identity;
            (identity, entry)
        })
        .collect::<BTreeMap<_, _>>();
    let source = CompiledPropertySource {
        geometry_id: compiled.identity.geometry_id.clone(),
        geometry_revision: compiled.identity.geometry_revision,
        mesh_digest: compiled.identity.mesh_digest,
        explicit_topology_digest: compiled.identity.explicit_topology_digest,
    };
    let element_bytes = encode_contract(
        SEMANTIC_ELEMENT_PREFIX,
        "compiled-property-semantics.solids",
        &solid_semantics(&template),
        ELEMENT_LIMITS,
    )
    .unwrap()
    .len() as u64;
    assert!(
        element_bytes * ELEMENTS > 512 * 1024 * 1024,
        "the fixture must exceed the retired v1 ceiling"
    );
    let empty = BTreeMap::new();
    let semantic = semantic_digest(&source, &solids, &empty, &BTreeMap::new(), &BTreeMap::new())
        .expect("a current-sensor-sized property map compiles its identity");
    audit_digest(
        semantic,
        &solids,
        &empty,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &[],
    )
    .expect("a current-sensor-sized property map compiles its audit identity");
    let mut changed = solids;
    changed.values_mut().last().unwrap().element_id += 1;
    assert_ne!(
        semantic,
        semantic_digest(
            &source,
            &changed,
            &empty,
            &BTreeMap::new(),
            &BTreeMap::new()
        )
        .unwrap(),
        "the streamed identity covers every element"
    );
}
