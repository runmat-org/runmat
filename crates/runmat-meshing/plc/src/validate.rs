use std::collections::{BTreeMap, BTreeSet};

pub use runmat_meshing_core::contracts::PlcValidationSummary;
use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TopologyEntityId};

pub const MODULE_PURPOSE: &str =
    "watertightness, manifold incidence, shell nesting, and material interfaces";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlcValidationError {
    ValidationSummaryNotVolumeReady {
        summary: PlcValidationSummary,
    },
    EmptyNodes,
    EmptyFacets,
    DuplicateNode {
        node_id: TopologyEntityId,
    },
    NonFiniteNode {
        node_id: TopologyEntityId,
    },
    FacetReferencesUnknownNode {
        facet_id: TopologyEntityId,
        node_id: TopologyEntityId,
    },
    FacetHasRepeatedNode {
        facet_id: TopologyEntityId,
    },
    ProtectedEdgeReferencesUnknownNode {
        edge_id: TopologyEntityId,
        node_id: TopologyEntityId,
    },
    ProtectedEdgeHasRepeatedNode {
        edge_id: TopologyEntityId,
    },
    UnreferencedNode {
        node_id: TopologyEntityId,
    },
    ProtectedEdgeNotOnBoundary {
        edge_id: TopologyEntityId,
        node_ids: [TopologyEntityId; 2],
    },
    OpenBoundaryEdge {
        node_ids: [TopologyEntityId; 2],
        incidence_count: usize,
    },
    NonManifoldBoundaryEdge {
        node_ids: [TopologyEntityId; 2],
        incidence_count: usize,
    },
}

impl std::fmt::Display for PlcValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValidationSummaryNotVolumeReady { .. } => {
                write!(formatter, "PLC validation summary is not volume-ready")
            }
            Self::EmptyNodes => write!(formatter, "PLC has no nodes"),
            Self::EmptyFacets => write!(formatter, "PLC has no facets"),
            Self::DuplicateNode { node_id } => {
                write!(formatter, "PLC has duplicate node {}", node_id.id)
            }
            Self::NonFiniteNode { node_id } => {
                write!(
                    formatter,
                    "PLC node {} has non-finite coordinates",
                    node_id.id
                )
            }
            Self::FacetReferencesUnknownNode { facet_id, node_id } => write!(
                formatter,
                "PLC facet {} references unknown node {}",
                facet_id.id, node_id.id
            ),
            Self::FacetHasRepeatedNode { facet_id } => {
                write!(formatter, "PLC facet {} repeats a node", facet_id.id)
            }
            Self::ProtectedEdgeReferencesUnknownNode { edge_id, node_id } => write!(
                formatter,
                "PLC protected edge {} references unknown node {}",
                edge_id.id, node_id.id
            ),
            Self::ProtectedEdgeHasRepeatedNode { edge_id } => {
                write!(
                    formatter,
                    "PLC protected edge {} repeats a node",
                    edge_id.id
                )
            }
            Self::UnreferencedNode { node_id } => {
                write!(
                    formatter,
                    "PLC node {} is not referenced by any facet",
                    node_id.id
                )
            }
            Self::ProtectedEdgeNotOnBoundary { edge_id, node_ids } => write!(
                formatter,
                "PLC protected edge {} references non-boundary edge {}-{}",
                edge_id.id, node_ids[0].id, node_ids[1].id
            ),
            Self::OpenBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has incidence {incidence_count}, expected 2",
                node_ids[0].id, node_ids[1].id
            ),
            Self::NonManifoldBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has non-manifold incidence {incidence_count}, expected 2",
                node_ids[0].id, node_ids[1].id
            ),
        }
    }
}

impl std::error::Error for PlcValidationError {}

pub fn validate_protected_boundary_complex(
    plc: &ProtectedBoundaryComplex,
) -> Result<PlcValidationSummary, PlcValidationError> {
    if !plc.validation.valid_for_volume_meshing() {
        return Err(PlcValidationError::ValidationSummaryNotVolumeReady {
            summary: plc.validation,
        });
    }
    if plc.nodes.is_empty() {
        return Err(PlcValidationError::EmptyNodes);
    }
    if plc.facets.is_empty() {
        return Err(PlcValidationError::EmptyFacets);
    }

    let mut node_ids = BTreeSet::<TopologyEntityId>::new();
    for node in &plc.nodes {
        if !node_ids.insert(node.node_id.clone()) {
            return Err(PlcValidationError::DuplicateNode {
                node_id: node.node_id.clone(),
            });
        }
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(PlcValidationError::NonFiniteNode {
                node_id: node.node_id.clone(),
            });
        }
    }

    let mut edge_incidence = BTreeMap::<[TopologyEntityId; 2], usize>::new();
    let mut referenced_node_ids = BTreeSet::<TopologyEntityId>::new();
    for facet in &plc.facets {
        validate_facet_nodes(facet.facet_id.clone(), facet.node_ids.as_ref(), &node_ids)?;
        referenced_node_ids.extend(facet.node_ids.iter().cloned());
        for edge_index in 0..3 {
            let edge = sorted_edge(
                facet.node_ids[edge_index].clone(),
                facet.node_ids[(edge_index + 1) % 3].clone(),
            );
            *edge_incidence.entry(edge).or_insert(0) += 1;
        }
    }

    for node_id in &node_ids {
        if !referenced_node_ids.contains(node_id) {
            return Err(PlcValidationError::UnreferencedNode {
                node_id: node_id.clone(),
            });
        }
    }

    for protected_edge in &plc.protected_edges {
        if protected_edge.node_ids[0] == protected_edge.node_ids[1] {
            return Err(PlcValidationError::ProtectedEdgeHasRepeatedNode {
                edge_id: protected_edge.edge_id.clone(),
            });
        }
        for node_id in &protected_edge.node_ids {
            if !node_ids.contains(node_id) {
                return Err(PlcValidationError::ProtectedEdgeReferencesUnknownNode {
                    edge_id: protected_edge.edge_id.clone(),
                    node_id: node_id.clone(),
                });
            }
        }
        let edge = sorted_edge(
            protected_edge.node_ids[0].clone(),
            protected_edge.node_ids[1].clone(),
        );
        if !edge_incidence.contains_key(&edge) {
            return Err(PlcValidationError::ProtectedEdgeNotOnBoundary {
                edge_id: protected_edge.edge_id.clone(),
                node_ids: edge,
            });
        }
    }

    for (edge, incidence_count) in edge_incidence {
        if incidence_count < 2 {
            return Err(PlcValidationError::OpenBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
        if incidence_count > 2 {
            return Err(PlcValidationError::NonManifoldBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
    }

    Ok(PlcValidationSummary {
        watertight: true,
        manifold: true,
        shell_nesting_classified: true,
        material_interfaces_classified: true,
    })
}

fn validate_facet_nodes(
    facet_id: TopologyEntityId,
    facet_node_ids: &[TopologyEntityId],
    known_node_ids: &BTreeSet<TopologyEntityId>,
) -> Result<(), PlcValidationError> {
    let mut unique_node_ids = BTreeSet::<TopologyEntityId>::new();
    for node_id in facet_node_ids {
        if !known_node_ids.contains(node_id) {
            return Err(PlcValidationError::FacetReferencesUnknownNode {
                facet_id,
                node_id: node_id.clone(),
            });
        }
        if !unique_node_ids.insert(node_id.clone()) {
            return Err(PlcValidationError::FacetHasRepeatedNode { facet_id });
        }
    }
    Ok(())
}

fn sorted_edge(left: TopologyEntityId, right: TopologyEntityId) -> [TopologyEntityId; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::contracts::{
        MeshingStage, PlcFacet, PlcNode, PlcProtectedEdge, ProtectedBoundaryComplex, StageEvidence,
    };

    #[test]
    fn validates_closed_manifold_plc() {
        let summary = validate_protected_boundary_complex(&tetrahedron_plc())
            .expect("closed manifold PLC should validate");

        assert!(summary.valid_for_volume_meshing());
    }

    #[test]
    fn rejects_not_volume_ready_validation_summary() {
        let mut plc = tetrahedron_plc();
        plc.validation.watertight = false;

        assert_eq!(
            validate_protected_boundary_complex(&plc),
            Err(PlcValidationError::ValidationSummaryNotVolumeReady {
                summary: plc.validation,
            })
        );
    }

    #[test]
    fn rejects_open_boundary_edge() {
        let mut plc = tetrahedron_plc();
        plc.facets.pop();

        assert!(matches!(
            validate_protected_boundary_complex(&plc),
            Err(PlcValidationError::OpenBoundaryEdge { .. })
        ));
    }

    #[test]
    fn rejects_facet_that_references_unknown_node() {
        let mut plc = tetrahedron_plc();
        plc.facets[0].node_ids[0] = entity("missing");

        assert!(matches!(
            validate_protected_boundary_complex(&plc),
            Err(PlcValidationError::FacetReferencesUnknownNode { .. })
        ));
    }

    #[test]
    fn rejects_unreferenced_node() {
        let mut plc = tetrahedron_plc();
        plc.nodes.push(node("4", [2.0, 2.0, 2.0]));

        assert!(matches!(
            validate_protected_boundary_complex(&plc),
            Err(PlcValidationError::UnreferencedNode { .. })
        ));
    }

    #[test]
    fn rejects_protected_edge_that_references_unknown_node() {
        let mut plc = tetrahedron_plc();
        plc.protected_edges.push(PlcProtectedEdge {
            edge_id: entity("edge_missing"),
            node_ids: [entity("0"), entity("missing")],
            source_edge_id: entity("source_edge"),
        });

        assert!(matches!(
            validate_protected_boundary_complex(&plc),
            Err(PlcValidationError::ProtectedEdgeReferencesUnknownNode { .. })
        ));
    }

    #[test]
    fn rejects_protected_edge_that_is_not_a_boundary_edge() {
        let mut plc = octahedron_plc();
        plc.protected_edges.push(PlcProtectedEdge {
            edge_id: entity("pole_to_pole"),
            node_ids: [entity("0"), entity("5")],
            source_edge_id: entity("source_edge"),
        });

        assert!(matches!(
            validate_protected_boundary_complex(&plc),
            Err(PlcValidationError::ProtectedEdgeNotOnBoundary { .. })
        ));
    }

    fn tetrahedron_plc() -> ProtectedBoundaryComplex {
        ProtectedBoundaryComplex {
            complex_id: "tetrahedron_plc".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 0.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [0.0, 1.0, 0.0]),
                node("3", [0.0, 0.0, 1.0]),
            ],
            facets: vec![
                facet("f0", ["0", "2", "1"]),
                facet("f1", ["0", "1", "3"]),
                facet("f2", ["1", "2", "3"]),
                facet("f3", ["2", "0", "3"]),
            ],
            protected_edges: Vec::new(),
            validation: PlcValidationSummary {
                watertight: true,
                manifold: true,
                shell_nesting_classified: true,
                material_interfaces_classified: true,
            },
            evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
        }
    }

    fn octahedron_plc() -> ProtectedBoundaryComplex {
        ProtectedBoundaryComplex {
            complex_id: "octahedron_plc".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 1.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [0.0, 1.0, 0.0]),
                node("3", [-1.0, 0.0, 0.0]),
                node("4", [0.0, -1.0, 0.0]),
                node("5", [0.0, 0.0, -1.0]),
            ],
            facets: vec![
                facet("f0", ["0", "1", "2"]),
                facet("f1", ["0", "2", "3"]),
                facet("f2", ["0", "3", "4"]),
                facet("f3", ["0", "4", "1"]),
                facet("f4", ["5", "2", "1"]),
                facet("f5", ["5", "3", "2"]),
                facet("f6", ["5", "4", "3"]),
                facet("f7", ["5", "1", "4"]),
            ],
            protected_edges: Vec::new(),
            validation: PlcValidationSummary {
                watertight: true,
                manifold: true,
                shell_nesting_classified: true,
                material_interfaces_classified: true,
            },
            evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
        }
    }

    fn node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
        PlcNode {
            node_id: entity(id),
            coordinates_m,
        }
    }

    fn facet(id: &str, node_ids: [&str; 3]) -> PlcFacet {
        PlcFacet {
            facet_id: entity(id),
            node_ids: [
                entity(node_ids[0]),
                entity(node_ids[1]),
                entity(node_ids[2]),
            ],
            source_face_id: entity(id),
            material_interface_ids: Vec::new(),
        }
    }

    fn entity(id: &str) -> TopologyEntityId {
        TopologyEntityId {
            stage: MeshingStage::ProtectedBoundaryComplex,
            id: id.to_string(),
        }
    }
}
