use super::*;

#[test]
fn recovery_queue_reports_missing_source_edge_embedded_in_interior_volume_edges() {
    let mesh = interior_source_edge_leak_mesh();

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("interior source edges should be reported as recovery evidence");

    assert_eq!(queue.evidence.entity_counts["missing_source_edge_items"], 1);
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_topology_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_interior_edge_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_source_edge_absent_edge_items"],
        0
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::InteriorEdge)
            && item.protected_edge_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                ])
    }));
}

#[test]
fn recovery_stage_result_removes_exterior_elements_across_interior_source_faces() {
    let result =
        recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), interior_source_edge_leak_mesh())
            .expect("outside elements across PLC facets should be removed before final audit");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(result.tetrahedron_mesh.elements.len(), 1);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["attempted_boundary_leak_recovery_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["interior_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["interior_face_source_face_recovery_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["removed_exterior_leaked_elements"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["exposed_interior_source_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["rejected_boundary_leak_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_face_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        2
    );
}

fn interior_source_edge_leak_mesh() -> TetrahedronMesh {
    let mut mesh = tetrahedron_mesh();
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::TetrahedronMesh, "interior_edge_node_1"),
        [0.0, 0.0, -1.0],
    ));
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::TetrahedronMesh, "interior_edge_node_2"),
        [0.0, -1.0, 0.0],
    ));
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "interior_edge_element_1"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            entity(MeshingStage::TetrahedronMesh, "interior_edge_node_1"),
        ],
        material_region_id: "solid_body".to_string(),
    });
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "interior_edge_element_2"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            entity(MeshingStage::TetrahedronMesh, "interior_edge_node_2"),
        ],
        material_region_id: "solid_body".to_string(),
    });
    mesh
}
