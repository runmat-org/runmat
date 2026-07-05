use std::collections::BTreeSet;

use runmat_meshing_core::contracts::{MeshingStage, TetrahedronMesh, TopologyEntityId};

use super::TetrahedronRecoveryError;

pub(super) fn validate_tetrahedron_recovery_input_mesh(
    tetrahedron_mesh: &TetrahedronMesh,
) -> Result<(), TetrahedronRecoveryError> {
    if tetrahedron_mesh.evidence.stage != MeshingStage::TetrahedronMesh {
        return Err(
            TetrahedronRecoveryError::TetrahedronMeshEvidenceStageMismatch {
                stage: tetrahedron_mesh.evidence.stage,
            },
        );
    }

    let mut node_ids = BTreeSet::<TopologyEntityId>::new();
    for node in &tetrahedron_mesh.nodes {
        if !node_ids.insert(node.node_id.clone()) {
            return Err(TetrahedronRecoveryError::DuplicateTetrahedronMeshNode {
                node_id: node.node_id.clone(),
            });
        }
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(TetrahedronRecoveryError::NonFiniteTetrahedronMeshNode {
                node_id: node.node_id.clone(),
            });
        }
    }

    let mut element_ids = BTreeSet::<TopologyEntityId>::new();
    for element in &tetrahedron_mesh.elements {
        if element.element_id.stage != MeshingStage::TetrahedronMesh {
            return Err(TetrahedronRecoveryError::TetrahedronElementStageMismatch {
                element_id: element.element_id.clone(),
            });
        }
        if !element_ids.insert(element.element_id.clone()) {
            return Err(TetrahedronRecoveryError::DuplicateTetrahedronElement {
                element_id: element.element_id.clone(),
            });
        }
        validate_node_references(
            &element.node_ids,
            &node_ids,
            |node_id| TetrahedronRecoveryError::TetrahedronElementReferencesUnknownNode {
                element_id: element.element_id.clone(),
                node_id,
            },
            || TetrahedronRecoveryError::TetrahedronElementHasRepeatedNode {
                element_id: element.element_id.clone(),
            },
        )?;
        if element.material_region_id.trim().is_empty() {
            return Err(
                TetrahedronRecoveryError::TetrahedronElementEmptyMaterialRegion {
                    element_id: element.element_id.clone(),
                },
            );
        }
    }

    let mut boundary_face_ids = BTreeSet::<TopologyEntityId>::new();
    for boundary_face in &tetrahedron_mesh.boundary_faces {
        if !matches!(
            boundary_face.face_id.stage,
            MeshingStage::ProtectedBoundaryComplex | MeshingStage::TetrahedronMesh
        ) {
            return Err(
                TetrahedronRecoveryError::TetrahedronBoundaryFaceStageMismatch {
                    face_id: boundary_face.face_id.clone(),
                },
            );
        }
        if !boundary_face_ids.insert(boundary_face.face_id.clone()) {
            return Err(TetrahedronRecoveryError::DuplicateTetrahedronBoundaryFace {
                face_id: boundary_face.face_id.clone(),
            });
        }
        validate_node_references(
            &boundary_face.node_ids,
            &node_ids,
            |node_id| TetrahedronRecoveryError::TetrahedronBoundaryFaceReferencesUnknownNode {
                face_id: boundary_face.face_id.clone(),
                node_id,
            },
            || TetrahedronRecoveryError::TetrahedronBoundaryFaceHasRepeatedNode {
                face_id: boundary_face.face_id.clone(),
            },
        )?;
        if boundary_face.source_face_id.stage != MeshingStage::SurfaceMesh {
            return Err(
                TetrahedronRecoveryError::TetrahedronBoundaryFaceSourceFaceStageMismatch {
                    face_id: boundary_face.face_id.clone(),
                    source_face_id: boundary_face.source_face_id.clone(),
                },
            );
        }
        for source_edge_id in boundary_face.source_edge_ids.iter().flatten() {
            if source_edge_id.stage != MeshingStage::CurveMesh {
                return Err(
                    TetrahedronRecoveryError::TetrahedronBoundaryFaceSourceEdgeStageMismatch {
                        face_id: boundary_face.face_id.clone(),
                        source_edge_id: source_edge_id.clone(),
                    },
                );
            }
        }
    }

    Ok(())
}

fn validate_node_references<const N: usize, UnknownNodeError, RepeatedNodeError>(
    node_ids: &[TopologyEntityId; N],
    known_node_ids: &BTreeSet<TopologyEntityId>,
    unknown_node_error: UnknownNodeError,
    repeated_node_error: RepeatedNodeError,
) -> Result<(), TetrahedronRecoveryError>
where
    UnknownNodeError: Fn(TopologyEntityId) -> TetrahedronRecoveryError,
    RepeatedNodeError: Fn() -> TetrahedronRecoveryError,
{
    let mut unique_node_ids = BTreeSet::<TopologyEntityId>::new();
    for node_id in node_ids {
        if !known_node_ids.contains(node_id) {
            return Err(unknown_node_error(node_id.clone()));
        }
        if !unique_node_ids.insert(node_id.clone()) {
            return Err(repeated_node_error());
        }
    }
    Ok(())
}
