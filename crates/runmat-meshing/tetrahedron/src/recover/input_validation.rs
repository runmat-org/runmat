use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, TetrahedronMesh, TopologyEntityId},
    quality::predicate::tetrahedron_signed_volume,
};

use super::{topology::sorted_topology_ids, TetrahedronRecoveryError};

const MIN_RECOVERY_INPUT_TETRAHEDRON_VOLUME_M3: f64 = 1.0e-18;

pub(super) fn validate_tetrahedron_recovery_input_mesh(
    plc: &ProtectedBoundaryComplex,
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
        if !matches!(
            node.node_id.stage,
            MeshingStage::ProtectedBoundaryComplex | MeshingStage::TetrahedronMesh
        ) {
            return Err(TetrahedronRecoveryError::TetrahedronMeshNodeStageMismatch {
                node_id: node.node_id.clone(),
            });
        }
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
    let node_coordinates = tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();

    let mut element_ids = BTreeSet::<TopologyEntityId>::new();
    let mut element_faces = BTreeSet::<[TopologyEntityId; 3]>::new();
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
        let signed_volume = tetrahedron_signed_volume(tetrahedron_element_points(
            element.node_ids.clone(),
            &node_coordinates,
        ));
        if !signed_volume.is_finite()
            || signed_volume.abs() <= MIN_RECOVERY_INPUT_TETRAHEDRON_VOLUME_M3
        {
            return Err(TetrahedronRecoveryError::DegenerateTetrahedronElement {
                element_id: element.element_id.clone(),
            });
        }
        element_faces.extend(tetrahedron_element_faces(element.node_ids.clone()));
    }

    let plc_faces = plc
        .facets
        .iter()
        .map(|facet| sorted_topology_ids(facet.node_ids.clone()))
        .collect::<BTreeSet<_>>();
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
        let boundary_face_key = sorted_topology_ids(boundary_face.node_ids.clone());
        if !element_faces.contains(&boundary_face_key) && !plc_faces.contains(&boundary_face_key) {
            return Err(
                TetrahedronRecoveryError::TetrahedronBoundaryFaceNotInElementOrPlcTopology {
                    face_id: boundary_face.face_id.clone(),
                },
            );
        }
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

fn tetrahedron_element_points(
    node_ids: [TopologyEntityId; 4],
    node_coordinates: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> [[f64; 3]; 4] {
    [
        node_coordinates[&node_ids[0]],
        node_coordinates[&node_ids[1]],
        node_coordinates[&node_ids[2]],
        node_coordinates[&node_ids[3]],
    ]
}

fn tetrahedron_element_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
    [
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
        sorted_topology_ids([
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ]),
    ]
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
