use runmat_meshing_core::contracts::{MeshingStage, StageEvidence, Tetrahedron4Element};

use super::{
    boundary_face, build_recovery_queue_from_plc, entity, tetrahedron_mesh, tetrahedron_node,
    tetrahedron_plc, TetrahedronRecoveryError,
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
fn accepts_recovery_input_with_generated_tetrahedron_boundary_entities() {
    let mut mesh = tetrahedron_mesh();
    let generated_node_id = entity(MeshingStage::TetrahedronMesh, "generated_boundary_node");
    mesh.nodes
        .push(tetrahedron_node(generated_node_id.clone(), [0.5, 0.0, 0.0]));
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "generated_element"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            generated_node_id.clone(),
            entity(MeshingStage::ProtectedBoundaryComplex, "3"),
        ],
        material_region_id: "solid_body".to_string(),
    });
    mesh.boundary_faces.push(boundary_face(
        "generated_boundary_face",
        ["0", "1", "3"],
        "face_2",
    ));
    mesh.boundary_faces
        .last_mut()
        .expect("generated boundary face should exist")
        .face_id = entity(MeshingStage::TetrahedronMesh, "generated_boundary_face");

    assert!(build_recovery_queue_from_plc(&tetrahedron_plc(), &mesh).is_ok());
}
