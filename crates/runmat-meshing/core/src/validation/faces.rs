use std::collections::BTreeSet;

use crate::artifact::AnalysisMeshArtifact;

use super::AnalysisMeshValidationError;

pub(super) fn validate_boundary_faces(
    mesh: &AnalysisMeshArtifact,
    node_ids: &BTreeSet<u32>,
    element_ids: &BTreeSet<String>,
) -> Result<BTreeSet<String>, AnalysisMeshValidationError> {
    let mut face_ids = BTreeSet::<String>::new();
    for face in &mesh.boundary_faces {
        if !face_ids.insert(face.face_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateBoundaryFaceId {
                face_id: face.face_id.clone(),
            });
        }
        if !face.kind.is_supported_for_boundary_mapping() {
            return Err(
                AnalysisMeshValidationError::UnsupportedBoundaryElementKind {
                    face_id: face.face_id.clone(),
                },
            );
        }
        let expected = face.kind.node_count();
        if face.node_ids.len() != expected {
            return Err(AnalysisMeshValidationError::WrongBoundaryFaceNodeCount {
                face_id: face.face_id.clone(),
                expected,
                actual: face.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &face.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownBoundaryFaceNode {
                    face_id: face.face_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedBoundaryFaceNode {
                    face_id: face.face_id.clone(),
                });
            }
        }
        for element_id in &face.adjacent_volume_element_ids {
            if !element_ids.contains(element_id) {
                return Err(
                    AnalysisMeshValidationError::UnknownBoundaryAdjacentElement {
                        face_id: face.face_id.clone(),
                        element_id: element_id.clone(),
                    },
                );
            }
        }
    }
    Ok(face_ids)
}
