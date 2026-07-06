use super::*;

#[test]
fn recovery_stage_result_repairs_single_material_interface_before_audit() {
    let mut mesh = tetrahedron_mesh();
    mesh.elements[0].material_region_id = "other_body".to_string();

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing material interface should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_material_interface_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("single material interface should repair element material ownership");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.tetrahedron_mesh.elements[0].material_region_id,
        "solid_body"
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_boundary_owned_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_material_interface_elements"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["attempted_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["global_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["interior_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["rejected_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
        0
    );
}

#[test]
fn recovery_stage_result_does_not_guess_multi_material_interface_repair() {
    let mut plc = tetrahedron_plc();
    plc.facets[0].material_interface_ids = vec!["other_body".to_string()];
    let mut mesh = tetrahedron_mesh();
    mesh.elements[0].material_region_id = "other_body".to_string();
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("ambiguous material interface should be reported before recovery");
    let mut direct_recovery_mesh = mesh.clone();
    let recovery = super::material_interfaces::recover_material_interface_regions(
        &plc,
        &initial_queue,
        &mut direct_recovery_mesh,
    );

    assert_eq!(recovery.attempted_material_interface_count, 1);
    assert_eq!(recovery.repaired_element_count, 0);
    assert_eq!(recovery.rejected_material_interface_count, 1);
    assert_eq!(recovery.global_material_interface_count, 0);
    assert_eq!(recovery.boundary_owned_material_interface_count, 1);
    assert_eq!(recovery.interior_material_interface_count, 0);
    assert_eq!(recovery.absent_partition_material_interface_count, 0);
    assert_eq!(recovery.ambiguous_boundary_ownership_count, 1);
    assert_eq!(recovery.missing_boundary_ownership_count, 0);
    assert_eq!(recovery.missing_interior_ownership_count, 0);
    assert_eq!(recovery.absent_partition_rejection_count, 0);

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh).expect_err("ambiguous repair should fail"),
        1,
        0,
        0,
        1,
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["boundary_owned_material_interface_recovery_input_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["absent_partition_material_interface_recovery_items"],
        0
    );
}

#[test]
fn recovery_stage_result_repairs_boundary_facet_owned_material_interface() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements[0].material_region_id = "region_b".to_string();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing material interface should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["material_interface_items"],
        2
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_material_interface_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("boundary-facet material ownership should repair the missing region");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.tetrahedron_mesh.elements[0].material_region_id,
        "region_a"
    );
    assert_eq!(
        result.tetrahedron_mesh.elements[1].material_region_id,
        "region_b"
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_boundary_owned_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_material_interface_elements"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["attempted_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["rejected_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["global_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["boundary_owned_material_interface_recovery_input_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["interior_material_interface_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
        0
    );
}

#[test]
fn recovery_queue_reports_incomplete_material_interface_ownership() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    add_unclassified_region_a_boundary_neighbor(&mut mesh);

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("incomplete material ownership should be reported before recovery");

    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_items"],
        1
    );
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.material_interface_id.as_deref() == Some("region_a")
            && item.material_interface_topology
                == Some(TetrahedronMaterialInterfaceTopology::BoundaryOwned)
    }));
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Recovered
            && item.material_interface_id.as_deref() == Some("region_b")
            && item.material_interface_topology.is_none()
    }));
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_boundary_owned_items"],
        1
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_interior_face_items"],
        0
    );
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_absent_partition_items"],
        0
    );
}

#[test]
fn recovery_queue_classifies_interior_face_material_interface_ownership() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    replace_region_b_with_unclassified_region_a_interior_neighbor(&mut mesh);

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("interior material ownership should be classified before recovery");

    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.material_interface_id.as_deref() == Some("region_a")
            && item.material_interface_topology
                == Some(TetrahedronMaterialInterfaceTopology::InteriorFace)
    }));
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_interior_face_items"],
        1
    );
}

#[test]
fn recovery_queue_classifies_absent_partition_material_interface_work() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));

    let queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent material partition should be classified before recovery");

    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::MaterialInterface
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.material_interface_id.as_deref() == Some("region_a")
            && item.material_interface_topology
                == Some(TetrahedronMaterialInterfaceTopology::AbsentPartition)
    }));
    assert_eq!(
        queue.evidence.entity_counts["missing_material_interface_absent_partition_items"],
        1
    );
}

#[test]
fn recovery_stage_result_inserts_bounded_absent_material_interface_partition() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("bounded absent material partition should be inserted");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result
        .tetrahedron_mesh
        .elements
        .iter()
        .any(|element| element.material_region_id == "region_a"));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_material_interface_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_face_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_partition_material_interface_recovery_input_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["inserted_absent_material_partition_elements"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_boundary_faces"],
        3
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_absent_partition_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        3
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_topology_candidate_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_usable_candidate_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_absent_material_partition_quality_candidate_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_absent_material_partition_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_material_interface_absent_partition"],
        0
    );
}

#[test]
fn recovery_stage_result_preserves_protected_source_edge_on_inserted_material_partition() {
    let plc = two_region_bipyramid_plc_with_region_a_protected_edge();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("bounded absent material partition should preserve protected edge provenance");

    let protected_source_edge_id = entity(MeshingStage::CurveMesh, "edge_region_a_0_2");
    assert!(result
        .tetrahedron_mesh
        .boundary_faces
        .iter()
        .filter(|face| face.source_face_id.id.starts_with("face_a"))
        .any(|face| face
            .source_edge_ids
            .iter()
            .any(|source_edge_id| source_edge_id.as_ref() == Some(&protected_source_edge_id))));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rolled_back_absent_material_partition_recovery_items"],
        0
    );
}

#[test]
fn recovery_stage_result_inserts_two_element_absent_material_interface_partition() {
    let plc = two_element_material_partition_plc();
    let mesh = two_element_material_partition_seed_mesh();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("two-element material partition should be queued before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_material_interface_absent_partition_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_face_absent_face_items"],
        6
    );

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("bounded two-element material partition should be inserted");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result
            .tetrahedron_mesh
            .elements
            .iter()
            .filter(|element| element.material_region_id == "region_a")
            .count(),
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["inserted_absent_material_partition_elements"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["inserted_absent_material_partition_boundary_faces"],
        6
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        6
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_absent_partition_material_interface_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_topology_candidate_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["absent_material_partition_usable_candidate_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_absent_material_partition_interior_candidate_sets"],
        2
    );
}

#[test]
fn recovery_stage_result_reports_absent_material_partition_quality_rejection() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));
    mesh.nodes
        .iter_mut()
        .find(|node| node.node_id.id == "4")
        .expect("fixture should carry apex node")
        .coordinates_m = [0.5, 0.5, 1.0e-12];

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh)
            .expect_err("degenerate material partition should fail the quality gate"),
        4,
        3,
        0,
        1,
    );

    assert_eq!(
        recovery_evidence.entity_counts["attempted_absent_material_partition_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["inserted_absent_material_partition_recovery_items"],
        0
    );
    assert_eq!(
        recovery_evidence.entity_counts["absent_material_partition_topology_candidate_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["absent_material_partition_usable_candidate_items"],
        0
    );
    assert_eq!(
        recovery_evidence.entity_counts
            ["rejected_absent_material_partition_quality_candidate_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_absent_material_partition_quality_gate"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_absent_material_partition_recovery_items"],
        1
    );
}

#[test]
fn recovery_stage_result_rejects_stale_absent_material_partition_boundary_topology() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));
    mesh.boundary_faces.push(boundary_face(
        "stale_facet_a_1",
        ["0", "2", "3"],
        "stale_face_a_1",
    ));

    assert_eq!(
        recover_tetrahedron_mesh_from_plc(&plc, mesh),
        Err(
            TetrahedronRecoveryError::TetrahedronBoundaryFaceNotInElementTopology {
                face_id: entity(MeshingStage::ProtectedBoundaryComplex, "stale_facet_a_1"),
            }
        )
    );
}

#[test]
fn recovery_stage_result_rejects_material_partition_with_stale_source_edge_boundary_topology() {
    let plc = two_region_bipyramid_plc_with_region_a_protected_edge();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    mesh.elements.remove(0);
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_b"));
    mesh.boundary_faces
        .push(boundary_face("facet_a_1", ["0", "2", "3"], "face_a_1"));

    assert_eq!(
        recover_tetrahedron_mesh_from_plc(&plc, mesh),
        Err(
            TetrahedronRecoveryError::TetrahedronBoundaryFaceNotInElementTopology {
                face_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_a_1"),
            }
        )
    );
}

#[test]
fn recovery_stage_result_rejects_material_repair_that_leaves_missing_source_face() {
    let plc = two_region_bipyramid_plc();
    let mut mesh = two_region_bipyramid_tetrahedron_mesh();
    add_unclassified_region_a_boundary_neighbor(&mut mesh);

    let recovery_evidence = assert_incomplete_recovery(
        recover_tetrahedron_mesh_from_plc(&plc, mesh)
            .expect_err("material ownership repair should not hide a missing source face"),
        1,
        1,
        0,
        0,
    );

    assert_eq!(
        recovery_evidence.entity_counts["missing_source_face_interior_face_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["attempted_boundary_leak_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_boundary_leak_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["rejected_boundary_leak_material_region_mismatch"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["removed_unsupported_boundary_faces"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["recovered_material_interface_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["repaired_material_interface_elements"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["attempted_material_interface_recovery_items"],
        1
    );
    assert_eq!(
        recovery_evidence.entity_counts["boundary_owned_material_interface_recovery_items"],
        1
    );
}

#[test]
fn material_interface_recovery_propagates_through_interior_faces() {
    let plc = interior_material_interface_propagation_plc();
    let initial_queue = TetrahedronRecoveryQueue {
        items: vec![TetrahedronRecoveryQueueItem {
            item_id: "material_interface:region_a".to_string(),
            kind: TetrahedronRecoveryKind::MaterialInterface,
            status: TetrahedronRecoveryStatus::Missing,
            source_entity_id: None,
            source_face_node_ids: None,
            source_face_topology: None,
            protected_edge_node_ids: None,
            protected_edge_topology: None,
            material_interface_topology: Some(TetrahedronMaterialInterfaceTopology::BoundaryOwned),
            material_interface_id: Some("region_a".to_string()),
        }],
        evidence: StageEvidence::complete(MeshingStage::ConstraintRecovery),
    };
    let mut mesh = interior_material_interface_propagation_mesh();

    let recovery = super::material_interfaces::recover_material_interface_regions(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(mesh.elements[0].material_region_id, "region_a");
    assert_eq!(mesh.elements[1].material_region_id, "region_a");
    assert_eq!(recovery.attempted_material_interface_count, 1);
    assert_eq!(recovery.repaired_element_count, 2);
    assert_eq!(recovery.rejected_material_interface_count, 0);
    assert_eq!(recovery.global_material_interface_count, 0);
    assert_eq!(recovery.boundary_owned_material_interface_count, 1);
    assert_eq!(recovery.interior_material_interface_count, 1);
    assert_eq!(recovery.absent_partition_material_interface_count, 0);
    assert_eq!(recovery.ambiguous_boundary_ownership_count, 0);
    assert_eq!(recovery.missing_boundary_ownership_count, 0);
    assert_eq!(recovery.missing_interior_ownership_count, 0);
    assert_eq!(recovery.absent_partition_rejection_count, 0);
}

#[test]
fn material_interface_recovery_uses_refined_boundary_source_face_ownership() {
    let plc = interior_material_interface_propagation_plc();
    let initial_queue = TetrahedronRecoveryQueue {
        items: vec![TetrahedronRecoveryQueueItem {
            item_id: "material_interface:region_a".to_string(),
            kind: TetrahedronRecoveryKind::MaterialInterface,
            status: TetrahedronRecoveryStatus::Missing,
            source_entity_id: None,
            source_face_node_ids: None,
            source_face_topology: None,
            protected_edge_node_ids: None,
            protected_edge_topology: None,
            material_interface_topology: Some(TetrahedronMaterialInterfaceTopology::BoundaryOwned),
            material_interface_id: Some("region_a".to_string()),
        }],
        evidence: StageEvidence::complete(MeshingStage::ConstraintRecovery),
    };
    let split_node_id = entity(MeshingStage::TetrahedronMesh, "split_face_a");
    let mut mesh = TetrahedronMesh {
        mesh_id: "refined_boundary_source_face_material_tetrahedron".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(split_node_id.clone(), [0.5, 0.5, 0.0]),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [0.0, 0.0, 1.0],
            ),
        ],
        elements: vec![Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_refined_face"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                split_node_id.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            material_region_id: "unclassified".to_string(),
        }],
        boundary_faces: vec![boundary_face_from_ids(
            "child_face_a",
            [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                split_node_id,
            ],
            "face_a",
            [None, None, None],
        )],
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    };

    let recovery = super::material_interfaces::recover_material_interface_regions(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(mesh.elements[0].material_region_id, "region_a");
    assert_eq!(recovery.attempted_material_interface_count, 1);
    assert_eq!(recovery.repaired_element_count, 1);
    assert_eq!(recovery.rejected_material_interface_count, 0);
    assert_eq!(recovery.boundary_owned_material_interface_count, 1);
    assert_eq!(recovery.interior_material_interface_count, 0);
}
