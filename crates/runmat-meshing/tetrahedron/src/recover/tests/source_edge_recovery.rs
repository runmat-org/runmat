use super::*;

#[test]
fn recovery_stage_result_records_protected_source_edge_recovered_by_boundary_faces() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.retain(|face| {
        !(face
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && face
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "1")))
    });

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing protected edge should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_topology_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        0
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_absent_edge_items"],
        0
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("missing protected edge should recover with its exterior boundary faces");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_missing_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_protected_edge_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["volume_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_volume_edge_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["attempted_source_edge_split_refill_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["accepted_source_edge_split_refill_candidate_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["rejected_source_edge_split_refill_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["applied_source_edge_split_refill_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["deferred_absent_source_edge_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn recovery_queue_accepts_split_boundary_face_chain_for_protected_source_edge() {
    let mut mesh = tetrahedron_mesh();
    let split_node = entity(MeshingStage::TetrahedronMesh, "source_edge_split_0_1");
    mesh.nodes
        .push(tetrahedron_node(split_node.clone(), [0.5, 0.0, 0.0]));
    mesh.elements = vec![
        Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "split_child_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                split_node.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            material_region_id: "solid_body".to_string(),
        },
        Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "split_child_2"),
            node_ids: [
                split_node.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            material_region_id: "solid_body".to_string(),
        },
    ];
    mesh.boundary_faces = vec![
        split_boundary_face(
            "split_face_1a",
            [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                split_node.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            ],
            "face_1",
        ),
        split_boundary_face(
            "split_face_1b",
            [
                split_node.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            ],
            "face_1",
        ),
        split_boundary_face(
            "split_face_2a",
            [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                split_node.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            "face_2",
        ),
        split_boundary_face(
            "split_face_2b",
            [
                split_node.clone(),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            "face_2",
        ),
        boundary_face("facet_3", ["1", "2", "3"], "face_3"),
        boundary_face("facet_4", ["2", "0", "3"], "face_4"),
    ];

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("split source-edge chain should satisfy recovery audit");

    assert_eq!(queue.evidence.entity_counts["missing_items"], 0);
    assert_eq!(queue.evidence.entity_counts["missing_source_edge_items"], 0);
    assert_eq!(queue.evidence.entity_counts["missing_source_face_items"], 0);
    assert!(queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Recovered
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::BoundaryEdge)
            && item.protected_edge_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                ])
    }));
    assert!(queue
        .items
        .iter()
        .filter(|item| item.kind == TetrahedronRecoveryKind::SourceFace)
        .all(|item| {
            item.status == TetrahedronRecoveryStatus::Recovered
                && item.source_face_topology == Some(TetrahedronSourceFaceTopology::BoundaryFace)
        }));
}

#[test]
fn source_edge_split_refill_application_closes_remaining_volume_edge_chain() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.retain(|face| {
        !(face
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && face
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "1")))
    });
    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing protected edge should be available for split/refill recovery");

    let recovery = crate::recover::source_edges::apply_source_edge_split_refill_recovery(
        &tetrahedron_plc(),
        &queue,
        &mut mesh,
    );

    assert_eq!(recovery.attempted_source_edge_count, 1);
    assert_eq!(recovery.accepted_source_edge_count, 1);
    assert_eq!(recovery.applied_source_edge_count, 1);
    assert_eq!(recovery.rejected_source_edge_count, 0);
    assert!(mesh.nodes.iter().any(|node| {
        node.node_id == entity(MeshingStage::TetrahedronMesh, "source_edge_split_0_1")
            && node.coordinates_m == [0.5, 0.0, 0.0]
    }));
    assert_eq!(mesh.elements.len(), 2);

    let recovered_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("split/refill recovery should leave an auditable mesh");
    assert_eq!(recovered_queue.evidence.entity_counts["missing_items"], 0);
    assert_eq!(
        recovered_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
    assert_eq!(
        recovered_queue.evidence.entity_counts["missing_source_face_items"],
        0
    );
}

fn split_boundary_face(
    id: &str,
    node_ids: [TopologyEntityId; 3],
    source_face_id: &str,
) -> TetrahedronBoundaryFace {
    TetrahedronBoundaryFace {
        face_id: entity(MeshingStage::TetrahedronMesh, id),
        source_edge_ids: split_source_edge_ids(node_ids.clone()),
        node_ids,
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
    }
}

fn split_source_edge_ids(node_ids: [TopologyEntityId; 3]) -> [Option<TopologyEntityId>; 3] {
    [
        split_source_edge_id_for_edge([node_ids[0].clone(), node_ids[1].clone()]),
        split_source_edge_id_for_edge([node_ids[1].clone(), node_ids[2].clone()]),
        split_source_edge_id_for_edge([node_ids[2].clone(), node_ids[0].clone()]),
    ]
}

fn split_source_edge_id_for_edge(mut node_ids: [TopologyEntityId; 2]) -> Option<TopologyEntityId> {
    node_ids.sort();
    let split_node = entity(MeshingStage::TetrahedronMesh, "source_edge_split_0_1");
    let mut left_child = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        split_node.clone(),
    ];
    let mut right_child = [
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        split_node,
    ];
    left_child.sort();
    right_child.sort();

    (node_ids == left_child || node_ids == right_child)
        .then(|| entity(MeshingStage::CurveMesh, "edge_1"))
}

#[test]
fn recovery_stage_result_records_cad_curve_source_edge_recovered_by_boundary_faces() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges[0].cad_curve_boundary = Some(cad_curve_boundary());
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.retain(|face| {
        !(face
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && face
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "1")))
    });

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("missing CAD curve protected edge should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_cad_curve_source_edge_topology_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("missing CAD curve protected edge should recover with boundary faces");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_cad_curve_protected_edge_boundary_face_restoration_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["recovered_cad_curve_protected_edge_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_cad_curve_source_edge_split_refill_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["accepted_cad_curve_source_edge_split_refill_candidate_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_cad_curve_source_edge_split_refill_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_cad_curve_protected_edge_boundary_face_restoration_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        0
    );
}

#[test]
fn protected_edge_boundary_restoration_reports_rejected_non_exterior_volume_face() {
    let plc = tetrahedron_plc();
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces.retain(|face| {
        !(face
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && face
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "1")))
    });
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::TetrahedronMesh, "face_support"),
        [0.25, 0.25, -1.0],
    ));
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "face_support_element"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            entity(MeshingStage::TetrahedronMesh, "face_support"),
        ],
        material_region_id: "solid_body".to_string(),
    });
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("volume-edge source edge should be reported before recovery");

    let restoration = crate::recover::boundary_faces::recover_missing_protected_edge_boundary_faces(
        &plc,
        &initial_queue,
        &mut mesh,
    );

    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        1
    );
    assert_eq!(restoration.attempted_boundary_face_count, 2);
    assert_eq!(restoration.recovered_boundary_face_count, 1);
    assert_eq!(restoration.rejected_boundary_face_count, 1);
    assert_eq!(
        restoration.rejection_counts["rejected_boundary_face_restoration_volume_face_topology"],
        1
    );
    assert!(mesh.boundary_faces.iter().any(|face| {
        sorted_face_ids(face.node_ids.clone())
            == sorted_face_ids([
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ])
    }));
    assert!(!mesh.boundary_faces.iter().any(|face| {
        sorted_face_ids(face.node_ids.clone())
            == sorted_face_ids([
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            ])
    }));
}

#[test]
fn recovery_stage_result_reconnects_absent_source_edge_by_boundary_diagonal_flip() {
    let plc = boundary_diagonal_flip_plc();
    let mesh = boundary_diagonal_flip_tetrahedron_mesh();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent protected edge should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_absent_edge_items"],
        1
    );
    assert!(initial_queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.status == TetrahedronRecoveryStatus::Missing
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::Absent)
            && item.protected_edge_node_ids
                == Some([
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                ])
    }));

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("boundary diagonal flip should recover the absent protected edge");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result.tetrahedron_mesh.elements.iter().any(|element| {
        element
            .node_ids
            .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "0"))
            && element
                .node_ids
                .contains(&entity(MeshingStage::ProtectedBoundaryComplex, "1"))
    }));
    assert!(result.tetrahedron_mesh.boundary_faces.iter().any(|face| {
        sorted_face_ids(face.node_ids.clone())
            == sorted_face_ids([
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            ])
            && face.source_edge_ids.iter().any(|source_edge_id| {
                source_edge_id
                    .as_ref()
                    .is_some_and(|source_edge_id| source_edge_id.id == "edge_1")
            })
    }));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["reconnected_absent_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["absent_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_absent_edge_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_absent_source_edge_boundary_faces"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["deferred_absent_source_edge_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_source_face_diagonal_recovery_pairs"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_face_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
}

#[test]
fn recovery_stage_result_reconnects_cad_curve_absent_source_edge_by_boundary_diagonal_flip() {
    let mut plc = boundary_diagonal_flip_plc();
    plc.protected_edges[0].cad_curve_boundary = Some(cad_curve_boundary());
    let mesh = boundary_diagonal_flip_tetrahedron_mesh();

    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent CAD curve protected edge should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_cad_curve_source_edge_topology_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("boundary diagonal flip should recover the CAD curve protected edge");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["attempted_cad_curve_absent_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["reconnected_cad_curve_absent_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["rejected_cad_curve_absent_source_edge_recovery_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        0
    );
}

#[test]
fn absent_source_edge_boundary_diagonal_flip_records_rejection_without_mutating_mesh() {
    let plc = boundary_diagonal_flip_plc();
    let mut mesh = boundary_diagonal_flip_tetrahedron_mesh();
    mesh.elements[1].material_region_id = "other_body".to_string();
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("absent protected edge should be reported before recovery");
    let original_elements = mesh.elements.clone();
    let original_boundary_faces = mesh.boundary_faces.clone();

    let recovery =
        crate::recover::absent_edges::recover_absent_protected_edges_by_boundary_diagonal_flip(
            &plc,
            &initial_queue,
            &mut mesh,
        );

    assert_eq!(recovery.attempted_source_edge_count, 1);
    assert_eq!(recovery.source_edge_count, 0);
    assert_eq!(recovery.boundary_face_count, 0);
    assert_eq!(recovery.rejected_source_edge_count, 1);
    assert_eq!(
        recovery.rejection_counts["rejected_absent_source_edge_recovery_material_region_mismatch"],
        1
    );
    assert_eq!(mesh.elements, original_elements);
    assert_eq!(mesh.boundary_faces, original_boundary_faces);
}

#[test]
fn recovery_stage_result_repairs_boundary_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    for boundary_face in &mut mesh.boundary_faces {
        boundary_face.source_edge_ids = [None, None, None];
    }

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("missing protected source-edge provenance should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_topology_items"],
        0
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        1
    );
    assert!(initial_queue.items.iter().any(|item| {
        item.kind == TetrahedronRecoveryKind::SourceEdge
            && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::BoundaryEdge)
    }));

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("matching boundary topology should repair source-edge provenance");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["boundary_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_boundary_edge_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn recovery_stage_result_repairs_cad_curve_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    for boundary_face in &mut mesh.boundary_faces {
        boundary_face.source_edge_ids = [None, None, None];
    }

    let initial_queue = build_recovery_queue_from_plc(&cad_curve_tetrahedron_plc(), &mesh)
        .expect("missing CAD curve source-edge provenance should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_cad_curve_source_edge_provenance_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&cad_curve_tetrahedron_plc(), mesh)
        .expect("matching boundary topology should repair CAD curve source-edge provenance");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["repaired_cad_curve_source_edge_provenance_items"],
        2
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_cad_curve_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_cad_curve_source_edge_items"],
        0
    );
}

#[test]
fn recovery_stage_result_repairs_partial_boundary_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_edge_ids[2] = None;

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("partial protected source-edge provenance should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_topology_items"],
        0
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("partial protected source-edge provenance should be repaired");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn recovery_stage_result_replaces_stale_boundary_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    for boundary_face in &mut mesh.boundary_faces {
        boundary_face.source_edge_ids = [
            Some(entity(MeshingStage::CurveMesh, "stale_edge")),
            Some(entity(MeshingStage::CurveMesh, "stale_edge")),
            Some(entity(MeshingStage::CurveMesh, "stale_edge")),
        ];
    }

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("stale protected source-edge provenance should be reported before recovery");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        1
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("matching boundary topology should replace stale source-edge provenance");

    let protected_edge = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
    ];
    for boundary_face in &result.tetrahedron_mesh.boundary_faces {
        for (edge_index, face_edge) in
            crate::protected_edges::face_edges(boundary_face.node_ids.clone())
                .into_iter()
                .enumerate()
        {
            if face_edge == protected_edge {
                assert_eq!(
                    boundary_face.source_edge_ids[edge_index],
                    Some(entity(MeshingStage::CurveMesh, "edge_1"))
                );
            } else {
                assert_eq!(boundary_face.source_edge_ids[edge_index], None);
            }
        }
    }
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        12
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["boundary_edge_source_edge_recovery_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}

#[test]
fn source_edge_provenance_repair_normalizes_boundary_slots_without_recovering_volume_edge() {
    let mut plc = tetrahedron_plc();
    plc.protected_edges[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
    ];
    plc.protected_edges[0].source_edge_id = entity(MeshingStage::CurveMesh, "edge_2");
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];
    mesh.boundary_faces[3].node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];
    let initial_queue = build_recovery_queue_from_plc(&plc, &mesh)
        .expect("volume-edge source edge should be reported before recovery");

    let repair =
        crate::recover::boundary_faces::repair_boundary_source_edge_provenance(&plc, &mut mesh);

    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_volume_edge_items"],
        1
    );
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_provenance_items"],
        0
    );
    assert_eq!(repair.repaired_count, 2);
    assert_eq!(repair.repaired_cad_curve_source_edge_count, 0);
    assert!(mesh.boundary_faces.iter().all(|face| {
        crate::protected_edges::face_edges(face.node_ids.clone())
            .into_iter()
            .enumerate()
            .all(|(edge_index, face_edge)| {
                let expected = (face_edge
                    == [
                        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    ])
                .then(|| entity(MeshingStage::CurveMesh, "edge_2"));
                face.source_edge_ids[edge_index] == expected
            })
    }));
}

#[test]
fn recovery_stage_result_removes_stale_non_protected_source_edge_provenance_before_audit() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[1].source_edge_ids[1] = Some(entity(MeshingStage::CurveMesh, "stale_edge"));

    let initial_queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("stale non-protected source-edge provenance should not hide recovered edge");
    assert_eq!(
        initial_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("stale non-protected source-edge provenance should be normalized");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(result.tetrahedron_mesh.boundary_faces.iter().all(|face| {
        crate::protected_edges::face_edges(face.node_ids.clone())
            .into_iter()
            .enumerate()
            .all(|(edge_index, face_edge)| {
                let expected = (face_edge
                    == [
                        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    ])
                .then(|| entity(MeshingStage::CurveMesh, "edge_1"));
                face.source_edge_ids[edge_index] == expected
            })
    }));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["repaired_source_edge_provenance_items"],
        1
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
}
