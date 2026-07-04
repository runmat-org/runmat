use runmat_meshing_core::contracts::{
    MeshingStage, StageEvidence, Tetrahedron4Element, TetrahedronBoundaryFace,
};

use super::{
    boundary_face, build_recovery_queue_from_plc, entity, recover_tetrahedron_mesh_from_plc,
    tetrahedron_mesh, tetrahedron_node, tetrahedron_plc, TetrahedronRecoveryError,
};

#[test]
fn rejects_recovery_input_with_non_tetrahedron_evidence_stage() {
    let mut mesh = tetrahedron_mesh();
    mesh.evidence = StageEvidence::complete(MeshingStage::ConstraintRecovery);

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(
            TetrahedronRecoveryError::TetrahedronMeshEvidenceStageMismatch {
                stage: MeshingStage::ConstraintRecovery,
            }
        )
    );
}

#[test]
fn rejects_recovery_input_with_duplicate_nodes() {
    let mut mesh = tetrahedron_mesh();
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        [0.0, 0.0, 0.0],
    ));

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(TetrahedronRecoveryError::DuplicateTetrahedronMeshNode {
            node_id: entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        })
    );
}

#[test]
fn rejects_recovery_input_with_non_tetrahedron_element_id() {
    let mut mesh = tetrahedron_mesh();
    mesh.elements[0].element_id = entity(MeshingStage::ProtectedBoundaryComplex, "wrong_stage");

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(TetrahedronRecoveryError::TetrahedronElementStageMismatch {
            element_id: entity(MeshingStage::ProtectedBoundaryComplex, "wrong_stage"),
        })
    );
}

#[test]
fn rejects_recovery_input_with_duplicate_element_id() {
    let mut mesh = tetrahedron_mesh();
    let mut duplicate_element = mesh.elements[0].clone();
    duplicate_element.node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        entity(MeshingStage::ProtectedBoundaryComplex, "0"),
        entity(MeshingStage::ProtectedBoundaryComplex, "2"),
        entity(MeshingStage::ProtectedBoundaryComplex, "3"),
    ];
    mesh.elements.push(duplicate_element);

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(TetrahedronRecoveryError::DuplicateTetrahedronElement {
            element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
        })
    );
}

#[test]
fn rejects_recovery_input_element_that_references_unknown_node() {
    let mut mesh = tetrahedron_mesh();
    mesh.elements[0].node_ids[0] = entity(MeshingStage::ProtectedBoundaryComplex, "missing");

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(
            TetrahedronRecoveryError::TetrahedronElementReferencesUnknownNode {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
                node_id: entity(MeshingStage::ProtectedBoundaryComplex, "missing"),
            },
        )
    );
}

#[test]
fn rejects_recovery_input_with_duplicate_boundary_face_id() {
    let mut mesh = tetrahedron_mesh();
    let mut duplicate_face = boundary_face("facet_1", ["0", "3", "1"], "face_2");
    duplicate_face.source_edge_ids = [None, None, None];
    mesh.boundary_faces.push(duplicate_face);

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(TetrahedronRecoveryError::DuplicateTetrahedronBoundaryFace {
            face_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
        })
    );
}

#[test]
fn rejects_recovery_input_boundary_face_with_bad_source_stages() {
    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_face_id = entity(MeshingStage::CurveMesh, "wrong_face_stage");

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(
            TetrahedronRecoveryError::TetrahedronBoundaryFaceSourceFaceStageMismatch {
                face_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
                source_face_id: entity(MeshingStage::CurveMesh, "wrong_face_stage"),
            },
        )
    );

    let mut mesh = tetrahedron_mesh();
    mesh.boundary_faces[0].source_edge_ids[0] =
        Some(entity(MeshingStage::SurfaceMesh, "wrong_edge_stage"));

    assert_eq!(
        build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh),
        Err(
            TetrahedronRecoveryError::TetrahedronBoundaryFaceSourceEdgeStageMismatch {
                face_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
                source_edge_id: entity(MeshingStage::SurfaceMesh, "wrong_edge_stage"),
            },
        )
    );
}

#[test]
fn ignores_and_removes_recovery_input_boundary_face_that_is_not_exterior() {
    let mut mesh = tetrahedron_mesh();
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "4"),
        [2.0, 2.0, 2.0],
    ));
    mesh.boundary_faces.push(boundary_face(
        "unsupported_boundary_face",
        ["0", "1", "4"],
        "face_2",
    ));

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("unsupported boundary face should not block queue classification");
    assert_eq!(queue.evidence.entity_counts["missing_items"], 0);

    let result = recover_tetrahedron_mesh_from_plc(&tetrahedron_plc(), mesh)
        .expect("unsupported boundary face should be removed before final audit");
    assert!(result.tetrahedron_mesh.recovery_complete);
    assert!(!result
        .tetrahedron_mesh
        .boundary_faces
        .iter()
        .any(|face| face.face_id.id == "unsupported_boundary_face"));
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["removed_unsupported_boundary_faces"],
        1
    );

    let mut mesh = tetrahedron_mesh();
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::TetrahedronMesh, "4"),
        [0.0, 0.0, -1.0],
    ));
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "interior_neighbor"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            entity(MeshingStage::TetrahedronMesh, "4"),
        ],
        material_region_id: "solid_body".to_string(),
    });

    let queue = build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh)
        .expect("interior boundary face should be classified as missing source-face work");
    assert_eq!(
        queue.evidence.entity_counts["missing_source_face_interior_face_items"],
        1
    );
}

#[test]
fn accepts_recovery_input_with_generated_tetrahedron_boundary_entities() {
    let mut mesh = tetrahedron_mesh();
    let generated_node_ids = ["generated_0", "generated_1", "generated_2", "generated_3"]
        .map(|id| entity(MeshingStage::TetrahedronMesh, id));
    for (node_id, coordinates_m) in generated_node_ids.clone().into_iter().zip([
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
    ]) {
        mesh.nodes.push(tetrahedron_node(node_id, coordinates_m));
    }
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "generated_element"),
        node_ids: generated_node_ids.clone(),
        material_region_id: "solid_body".to_string(),
    });
    mesh.boundary_faces.push(TetrahedronBoundaryFace {
        face_id: entity(MeshingStage::TetrahedronMesh, "generated_boundary_face"),
        node_ids: [
            generated_node_ids[0].clone(),
            generated_node_ids[1].clone(),
            generated_node_ids[2].clone(),
        ],
        source_face_id: entity(MeshingStage::SurfaceMesh, "generated_source_face"),
        source_edge_ids: [None, None, None],
    });

    assert!(build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh).is_ok());
}
