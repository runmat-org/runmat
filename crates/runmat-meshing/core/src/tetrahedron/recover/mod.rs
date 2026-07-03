pub mod boundary_queue;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::contracts::{
    MeshingStage, ProtectedBoundaryComplex, StageEvidence, TetrahedronMesh, TopologyEntityId,
};

pub const MODULE_PURPOSE: &str = "source-edge, source-face, and material-interface recovery queues";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetrahedronRecoveryError {
    InvalidProtectedBoundaryComplex,
    EmptyTetrahedronMesh,
    MissingSourceFaceRecovery { face_id: String },
    MissingSourceEdgeRecovery { edge_id: String },
    MissingMaterialInterfaceRecovery { material_interface_id: String },
}

impl std::fmt::Display for TetrahedronRecoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtectedBoundaryComplex => {
                write!(formatter, "Tetrahedron recovery requires a validated PLC")
            }
            Self::EmptyTetrahedronMesh => write!(
                formatter,
                "Tetrahedron recovery requires a non-empty Tetrahedron mesh"
            ),
            Self::MissingSourceFaceRecovery { face_id } => {
                write!(formatter, "source face {face_id} is not recovered")
            }
            Self::MissingSourceEdgeRecovery { edge_id } => {
                write!(formatter, "source edge {edge_id} is not recovered")
            }
            Self::MissingMaterialInterfaceRecovery {
                material_interface_id,
            } => write!(
                formatter,
                "material interface {material_interface_id} is not recovered"
            ),
        }
    }
}

impl std::error::Error for TetrahedronRecoveryError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronRecoveryQueue {
    #[serde(default)]
    pub items: Vec<TetrahedronRecoveryQueueItem>,
    pub evidence: StageEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronRecoveryQueueItem {
    pub item_id: String,
    pub kind: TetrahedronRecoveryKind,
    pub status: TetrahedronRecoveryStatus,
    #[serde(default)]
    pub source_entity_id: Option<TopologyEntityId>,
    #[serde(default)]
    pub material_interface_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronRecoveryKind {
    SourceFace,
    SourceEdge,
    MaterialInterface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronRecoveryStatus {
    Recovered,
}

pub fn build_recovery_queue_from_plc(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &TetrahedronMesh,
) -> Result<TetrahedronRecoveryQueue, TetrahedronRecoveryError> {
    if !plc.validation.valid_for_volume_meshing() {
        return Err(TetrahedronRecoveryError::InvalidProtectedBoundaryComplex);
    }
    if tetrahedron_mesh.nodes.is_empty() || tetrahedron_mesh.elements.is_empty() {
        return Err(TetrahedronRecoveryError::EmptyTetrahedronMesh);
    }

    let recovered_face_keys = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            (
                face.source_face_id.clone(),
                sorted_topology_ids(face.node_ids.clone()),
            )
        })
        .collect::<BTreeSet<_>>();
    let recovered_boundary_edges = tetrahedron_mesh
        .boundary_faces
        .iter()
        .flat_map(|face| topology_face_edges(face.node_ids.clone()))
        .collect::<BTreeSet<_>>();
    let recovered_material_interfaces = tetrahedron_mesh
        .elements
        .iter()
        .map(|element| element.material_region_id.clone())
        .collect::<BTreeSet<_>>();

    let mut items = Vec::<TetrahedronRecoveryQueueItem>::new();
    for facet in &plc.facets {
        let face_key = (
            facet.source_face_id.clone(),
            sorted_topology_ids(facet.node_ids.clone()),
        );
        if !recovered_face_keys.contains(&face_key) {
            return Err(TetrahedronRecoveryError::MissingSourceFaceRecovery {
                face_id: facet.source_face_id.id.clone(),
            });
        }
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_face:{}", facet.facet_id.id),
            kind: TetrahedronRecoveryKind::SourceFace,
            status: TetrahedronRecoveryStatus::Recovered,
            source_entity_id: Some(facet.source_face_id.clone()),
            material_interface_id: None,
        });
    }

    for protected_edge in &plc.protected_edges {
        let edge_key = sorted_topology_ids(protected_edge.node_ids.clone());
        if !recovered_boundary_edges.contains(&edge_key) {
            return Err(TetrahedronRecoveryError::MissingSourceEdgeRecovery {
                edge_id: protected_edge.source_edge_id.id.clone(),
            });
        }
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("source_edge:{}", protected_edge.edge_id.id),
            kind: TetrahedronRecoveryKind::SourceEdge,
            status: TetrahedronRecoveryStatus::Recovered,
            source_entity_id: Some(protected_edge.source_edge_id.clone()),
            material_interface_id: None,
        });
    }

    let material_interfaces = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    for material_interface_id in material_interfaces {
        if !recovered_material_interfaces.contains(&material_interface_id) {
            return Err(TetrahedronRecoveryError::MissingMaterialInterfaceRecovery {
                material_interface_id,
            });
        }
        items.push(TetrahedronRecoveryQueueItem {
            item_id: format!("material_interface:{material_interface_id}"),
            kind: TetrahedronRecoveryKind::MaterialInterface,
            status: TetrahedronRecoveryStatus::Recovered,
            source_entity_id: None,
            material_interface_id: Some(material_interface_id),
        });
    }

    let mut evidence = StageEvidence::complete(MeshingStage::ConstraintRecovery);
    evidence
        .entity_counts
        .insert("recovery_items".to_string(), items.len());
    evidence.entity_counts.insert(
        "source_face_items".to_string(),
        items
            .iter()
            .filter(|item| item.kind == TetrahedronRecoveryKind::SourceFace)
            .count(),
    );
    evidence.entity_counts.insert(
        "source_edge_items".to_string(),
        items
            .iter()
            .filter(|item| item.kind == TetrahedronRecoveryKind::SourceEdge)
            .count(),
    );
    evidence.entity_counts.insert(
        "material_interface_items".to_string(),
        items
            .iter()
            .filter(|item| item.kind == TetrahedronRecoveryKind::MaterialInterface)
            .count(),
    );

    Ok(TetrahedronRecoveryQueue { items, evidence })
}

fn topology_face_edges(node_ids: [TopologyEntityId; 3]) -> [[TopologyEntityId; 2]; 3] {
    [
        sorted_topology_ids([node_ids[0].clone(), node_ids[1].clone()]),
        sorted_topology_ids([node_ids[1].clone(), node_ids[2].clone()]),
        sorted_topology_ids([node_ids[2].clone(), node_ids[0].clone()]),
    ]
}

fn sorted_topology_ids<const N: usize>(
    mut node_ids: [TopologyEntityId; N],
) -> [TopologyEntityId; N] {
    node_ids.sort_by(|left, right| {
        topology_stage_rank(left.stage)
            .cmp(&topology_stage_rank(right.stage))
            .then_with(|| left.id.cmp(&right.id))
    });
    node_ids
}

fn topology_stage_rank(stage: MeshingStage) -> u8 {
    match stage {
        MeshingStage::CadTopology => 0,
        MeshingStage::Sizing => 1,
        MeshingStage::CurveMesh => 2,
        MeshingStage::SurfaceMesh => 3,
        MeshingStage::ProtectedBoundaryComplex => 4,
        MeshingStage::TetrahedronMesh => 5,
        MeshingStage::ConstraintRecovery => 6,
        MeshingStage::Optimization => 7,
        MeshingStage::SolveReadiness => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{
        PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, Tetrahedron4Element,
        TetrahedronBoundaryFace, TetrahedronMeshNode,
    };

    #[test]
    fn builds_recovery_queue_for_recovered_plc_constraints() {
        let queue =
            build_recovery_queue_from_plc(&single_facet_plc(), &single_facet_tetrahedron_mesh())
                .expect("matching Tetrahedron mesh should recover PLC constraints");

        assert_eq!(queue.items.len(), 3);
        assert_eq!(queue.evidence.stage, MeshingStage::ConstraintRecovery);
        assert_eq!(queue.evidence.entity_counts["source_face_items"], 1);
        assert_eq!(queue.evidence.entity_counts["source_edge_items"], 1);
        assert_eq!(queue.evidence.entity_counts["material_interface_items"], 1);
        assert!(queue
            .items
            .iter()
            .all(|item| item.status == TetrahedronRecoveryStatus::Recovered));
    }

    #[test]
    fn recovery_queue_rejects_missing_source_face() {
        let mut mesh = single_facet_tetrahedron_mesh();
        mesh.boundary_faces[0].source_face_id = entity(MeshingStage::SurfaceMesh, "other");

        assert_eq!(
            build_recovery_queue_from_plc(&single_facet_plc(), &mesh),
            Err(TetrahedronRecoveryError::MissingSourceFaceRecovery {
                face_id: "face_1".to_string()
            })
        );
    }

    #[test]
    fn recovery_queue_rejects_missing_source_edge() {
        let mut plc = single_facet_plc();
        plc.protected_edges[0].node_ids = [
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            entity(MeshingStage::ProtectedBoundaryComplex, "3"),
        ];

        assert_eq!(
            build_recovery_queue_from_plc(&plc, &single_facet_tetrahedron_mesh()),
            Err(TetrahedronRecoveryError::MissingSourceEdgeRecovery {
                edge_id: "edge_1".to_string()
            })
        );
    }

    #[test]
    fn recovery_queue_rejects_missing_material_interface() {
        let mut mesh = single_facet_tetrahedron_mesh();
        mesh.elements[0].material_region_id = "other_body".to_string();

        assert_eq!(
            build_recovery_queue_from_plc(&single_facet_plc(), &mesh),
            Err(TetrahedronRecoveryError::MissingMaterialInterfaceRecovery {
                material_interface_id: "solid_body".to_string()
            })
        );
    }

    fn single_facet_plc() -> ProtectedBoundaryComplex {
        ProtectedBoundaryComplex {
            complex_id: "single_facet_plc".to_string(),
            nodes: vec![
                plc_node("0", [0.0, 0.0, 0.0]),
                plc_node("1", [1.0, 0.0, 0.0]),
                plc_node("2", [0.0, 1.0, 0.0]),
            ],
            facets: vec![PlcFacet {
                facet_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                ],
                source_face_id: entity(MeshingStage::SurfaceMesh, "face_1"),
                material_interface_ids: vec!["solid_body".to_string()],
            }],
            protected_edges: vec![PlcProtectedEdge {
                edge_id: entity(MeshingStage::ProtectedBoundaryComplex, "plc_edge_1"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                ],
                source_edge_id: entity(MeshingStage::CurveMesh, "edge_1"),
            }],
            validation: PlcValidationSummary {
                watertight: true,
                manifold: true,
                shell_nesting_classified: true,
                material_interfaces_classified: true,
            },
            evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
        }
    }

    fn single_facet_tetrahedron_mesh() -> TetrahedronMesh {
        TetrahedronMesh {
            mesh_id: "single_facet_tetrahedron".to_string(),
            nodes: vec![
                tetrahedron_node(
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    [0.0, 0.0, 0.0],
                ),
                tetrahedron_node(
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    [1.0, 0.0, 0.0],
                ),
                tetrahedron_node(
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    [0.0, 1.0, 0.0],
                ),
                tetrahedron_node(entity(MeshingStage::TetrahedronMesh, "3"), [0.0, 0.0, 1.0]),
            ],
            elements: vec![Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::TetrahedronMesh, "3"),
                ],
                material_region_id: "solid_body".to_string(),
            }],
            boundary_faces: vec![TetrahedronBoundaryFace {
                face_id: entity(MeshingStage::ProtectedBoundaryComplex, "facet_1"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                ],
                source_face_id: entity(MeshingStage::SurfaceMesh, "face_1"),
            }],
            recovery_complete: false,
            quality_optimized: false,
            evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
        }
    }

    fn plc_node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
        PlcNode {
            node_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
            coordinates_m,
        }
    }

    fn tetrahedron_node(node_id: TopologyEntityId, coordinates_m: [f64; 3]) -> TetrahedronMeshNode {
        TetrahedronMeshNode {
            node_id,
            coordinates_m,
        }
    }

    fn entity(stage: MeshingStage, id: &str) -> TopologyEntityId {
        TopologyEntityId {
            stage,
            id: id.to_string(),
        }
    }
}
