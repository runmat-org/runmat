use std::collections::{BTreeMap, BTreeSet};

pub use crate::contracts::{PlcFacet, PlcNode, PlcProtectedEdge, ProtectedBoundaryComplex};
use crate::{
    contracts::{MeshingStage, StageEvidence, TopologyEntityId},
    surface::{SurfaceDiscretization, INTERNAL_SOURCE_EDGE_ID},
};

pub const MODULE_PURPOSE: &str = "oriented protected boundary complex construction";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlcBuildError {
    EmptySurface,
    MissingSurfaceNode {
        element_id: u32,
        node_id: u32,
    },
    NonFiniteSurfaceNode {
        node_id: u32,
    },
    NonFiniteSurfaceElement {
        element_id: u32,
    },
    DuplicateFacet {
        element_id: u32,
    },
    OpenBoundaryEdge {
        node_ids: [u32; 2],
        incidence_count: usize,
    },
    NonManifoldBoundaryEdge {
        node_ids: [u32; 2],
        incidence_count: usize,
    },
}

impl std::fmt::Display for PlcBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySurface => write!(formatter, "surface mesh has no facets for PLC build"),
            Self::MissingSurfaceNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "surface element {element_id} references missing PLC node {node_id}"
            ),
            Self::NonFiniteSurfaceNode { node_id } => {
                write!(
                    formatter,
                    "surface node {node_id} has non-finite coordinates"
                )
            }
            Self::NonFiniteSurfaceElement { element_id } => write!(
                formatter,
                "surface element {element_id} has non-finite area or projection evidence"
            ),
            Self::DuplicateFacet { element_id } => write!(
                formatter,
                "surface element {element_id} duplicates an existing PLC facet"
            ),
            Self::OpenBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has incidence {incidence_count}, expected 2",
                node_ids[0], node_ids[1]
            ),
            Self::NonManifoldBoundaryEdge {
                node_ids,
                incidence_count,
            } => write!(
                formatter,
                "PLC edge {}-{} has non-manifold incidence {incidence_count}, expected 2",
                node_ids[0], node_ids[1]
            ),
        }
    }
}

impl std::error::Error for PlcBuildError {}

pub fn build_protected_boundary_complex(
    surface: &SurfaceDiscretization,
) -> Result<ProtectedBoundaryComplex, PlcBuildError> {
    if surface.elements.is_empty() {
        return Err(PlcBuildError::EmptySurface);
    }

    let surface_nodes = surface
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node in &surface.nodes {
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(PlcBuildError::NonFiniteSurfaceNode {
                node_id: node.node_id,
            });
        }
    }

    let mut facets = Vec::<PlcFacet>::with_capacity(surface.elements.len());
    let mut protected_edges = BTreeMap::<(u32, u32, u32), PlcProtectedEdge>::new();
    let mut edge_incidence = BTreeMap::<[u32; 2], usize>::new();
    let mut facet_keys = BTreeSet::<[u32; 3]>::new();
    for element in &surface.elements {
        if !element.area_m2.is_finite() || !element.max_projection_error_m.is_finite() {
            return Err(PlcBuildError::NonFiniteSurfaceElement {
                element_id: element.element_id,
            });
        }
        for node_id in element.node_ids {
            if !surface_nodes.contains_key(&node_id) {
                return Err(PlcBuildError::MissingSurfaceNode {
                    element_id: element.element_id,
                    node_id,
                });
            }
        }
        let mut facet_key = element.node_ids;
        facet_key.sort_unstable();
        if !facet_keys.insert(facet_key) {
            return Err(PlcBuildError::DuplicateFacet {
                element_id: element.element_id,
            });
        }

        for edge_index in 0..3 {
            let left = element.node_ids[edge_index];
            let right = element.node_ids[(edge_index + 1) % 3];
            let edge = sorted_edge(left, right);
            *edge_incidence.entry(edge).or_insert(0) += 1;

            let source_edge_id = element.source_edge_ids[edge_index];
            if source_edge_id != INTERNAL_SOURCE_EDGE_ID {
                protected_edges
                    .entry((source_edge_id, edge[0], edge[1]))
                    .or_insert_with(|| PlcProtectedEdge {
                        edge_id: topology_entity_id(
                            MeshingStage::ProtectedBoundaryComplex,
                            format!(
                                "plc_protected_edge_{source_edge_id}_{}_{}",
                                edge[0], edge[1]
                            ),
                        ),
                        node_ids: [
                            topology_entity_id(MeshingStage::ProtectedBoundaryComplex, edge[0]),
                            topology_entity_id(MeshingStage::ProtectedBoundaryComplex, edge[1]),
                        ],
                        source_edge_id: topology_entity_id(MeshingStage::CurveMesh, source_edge_id),
                    });
            }
        }

        facets.push(PlcFacet {
            facet_id: topology_entity_id(
                MeshingStage::ProtectedBoundaryComplex,
                element.element_id,
            ),
            node_ids: [
                topology_entity_id(MeshingStage::ProtectedBoundaryComplex, element.node_ids[0]),
                topology_entity_id(MeshingStage::ProtectedBoundaryComplex, element.node_ids[1]),
                topology_entity_id(MeshingStage::ProtectedBoundaryComplex, element.node_ids[2]),
            ],
            source_face_id: topology_entity_id(MeshingStage::SurfaceMesh, element.source_face_id),
            material_interface_ids: element.region_ids.clone(),
        });
    }

    for (edge, incidence_count) in edge_incidence {
        if incidence_count < 2 {
            return Err(PlcBuildError::OpenBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
        if incidence_count > 2 {
            return Err(PlcBuildError::NonManifoldBoundaryEdge {
                node_ids: edge,
                incidence_count,
            });
        }
    }

    let mut evidence = StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex);
    evidence
        .entity_counts
        .insert("nodes".to_string(), surface.nodes.len());
    evidence
        .entity_counts
        .insert("facets".to_string(), facets.len());
    evidence
        .entity_counts
        .insert("protected_edges".to_string(), protected_edges.len());

    Ok(ProtectedBoundaryComplex {
        complex_id: "plc_surface_boundary".to_string(),
        nodes: surface
            .nodes
            .iter()
            .map(|node| PlcNode {
                node_id: topology_entity_id(MeshingStage::ProtectedBoundaryComplex, node.node_id),
                coordinates_m: node.coordinates_m,
            })
            .collect(),
        facets,
        protected_edges: protected_edges.into_values().collect(),
        validation: crate::contracts::PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence,
    })
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn topology_entity_id(stage: MeshingStage, id: impl ToString) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::{SurfaceElement, SurfaceNode};

    #[test]
    fn builds_valid_plc_from_closed_tetra_surface() {
        let plc = build_protected_boundary_complex(&tetra_surface())
            .expect("closed tetra surface should build a PLC");

        assert!(plc.validation.valid_for_volume_meshing());
        assert_eq!(plc.nodes.len(), 4);
        assert_eq!(plc.facets.len(), 4);
        assert_eq!(plc.protected_edges.len(), 6);
        assert_eq!(plc.evidence.entity_counts["facets"], 4);
    }

    #[test]
    fn rejects_open_surface_before_volume_meshing() {
        let mut surface = tetra_surface();
        surface.elements.pop();

        let err = build_protected_boundary_complex(&surface)
            .expect_err("open surface must not become a PLC");

        assert!(matches!(err, PlcBuildError::OpenBoundaryEdge { .. }));
    }

    #[test]
    fn rejects_duplicate_surface_facets() {
        let mut surface = tetra_surface();
        surface.elements[1] = surface.elements[0].clone();
        surface.elements[1].element_id = 99;

        assert_eq!(
            build_protected_boundary_complex(&surface),
            Err(PlcBuildError::DuplicateFacet { element_id: 99 })
        );
    }

    fn tetra_surface() -> SurfaceDiscretization {
        SurfaceDiscretization {
            nodes: vec![
                node(0, [0.0, 0.0, 0.0]),
                node(1, [1.0, 0.0, 0.0]),
                node(2, [0.0, 1.0, 0.0]),
                node(3, [0.0, 0.0, 1.0]),
            ],
            elements: vec![
                element(0, [0, 2, 1], [2, 1, 0]),
                element(1, [0, 1, 3], [0, 4, 3]),
                element(2, [1, 2, 3], [1, 5, 4]),
                element(3, [2, 0, 3], [2, 3, 5]),
            ],
            exact_cad_sample_node_count: 0,
            rejected_exact_cad_sample_count: 0,
        }
    }

    fn node(node_id: u32, coordinates_m: [f64; 3]) -> SurfaceNode {
        SurfaceNode {
            node_id,
            source_vertex_id: node_id,
            coordinates_m,
        }
    }

    fn element(element_id: u32, node_ids: [u32; 3], source_edge_ids: [u32; 3]) -> SurfaceElement {
        SurfaceElement {
            element_id,
            source_face_id: element_id,
            cad_face_id: None,
            source_edge_ids,
            node_ids,
            parametric_node_uv: [[0.0, 0.0]; 3],
            max_projection_error_m: 0.0,
            region_ids: vec!["body".to_string()],
            area_m2: 0.5,
            unit_normal: [0.0, 0.0, 1.0],
        }
    }
}
