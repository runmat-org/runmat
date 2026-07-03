use serde::{Deserialize, Serialize};

use runmat_meshing_cad::{SourceTopologyEdge, SourceTopologyModel};

pub const MODULE_PURPOSE: &str = "CAD edge discretization before surface or volume meshing";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurveDiscretizationOptions {
    pub target_size_m: f64,
    pub min_segments_per_edge: usize,
    pub max_segments_per_edge: usize,
}

impl Default for CurveDiscretizationOptions {
    fn default() -> Self {
        Self {
            target_size_m: 0.05,
            min_segments_per_edge: 1,
            max_segments_per_edge: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveNode {
    pub node_id: u32,
    pub source_edge_id: u32,
    pub parameter: f64,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveElement {
    pub element_id: u32,
    pub source_edge_id: u32,
    pub node_ids: [u32; 2],
    pub length_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveDiscretization {
    pub nodes: Vec<CurveNode>,
    pub elements: Vec<CurveElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveDiscretizationError {
    InvalidTargetSize,
    InvalidSegmentBounds,
    MissingEdgeVertex { edge_id: u32, node_id: u32 },
}

impl std::fmt::Display for CurveDiscretizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTargetSize => {
                write!(formatter, "curve target_size_m must be finite and positive")
            }
            Self::InvalidSegmentBounds => write!(
                formatter,
                "curve segment bounds must satisfy 1 <= min_segments_per_edge <= max_segments_per_edge"
            ),
            Self::MissingEdgeVertex { edge_id, node_id } => write!(
                formatter,
                "source edge {edge_id} references missing topology vertex {node_id}"
            ),
        }
    }
}

impl std::error::Error for CurveDiscretizationError {}

pub fn discretize_topology_curves(
    topology: &SourceTopologyModel,
    options: CurveDiscretizationOptions,
) -> Result<CurveDiscretization, CurveDiscretizationError> {
    validate_curve_options(options)?;
    let mut nodes = Vec::<CurveNode>::new();
    let mut elements = Vec::<CurveElement>::new();

    for edge in &topology.edges {
        let left = topology
            .vertices
            .get(edge.node_ids[0] as usize)
            .filter(|vertex| vertex.vertex_id == edge.node_ids[0])
            .map(|vertex| vertex.coordinates_m)
            .ok_or(CurveDiscretizationError::MissingEdgeVertex {
                edge_id: edge.edge_id,
                node_id: edge.node_ids[0],
            })?;
        let right = topology
            .vertices
            .get(edge.node_ids[1] as usize)
            .filter(|vertex| vertex.vertex_id == edge.node_ids[1])
            .map(|vertex| vertex.coordinates_m)
            .ok_or(CurveDiscretizationError::MissingEdgeVertex {
                edge_id: edge.edge_id,
                node_id: edge.node_ids[1],
            })?;
        append_edge_discretization(edge, left, right, options, &mut nodes, &mut elements);
    }

    Ok(CurveDiscretization { nodes, elements })
}

fn validate_curve_options(
    options: CurveDiscretizationOptions,
) -> Result<(), CurveDiscretizationError> {
    if !options.target_size_m.is_finite() || options.target_size_m <= 0.0 {
        return Err(CurveDiscretizationError::InvalidTargetSize);
    }
    if options.min_segments_per_edge == 0
        || options.min_segments_per_edge > options.max_segments_per_edge
    {
        return Err(CurveDiscretizationError::InvalidSegmentBounds);
    }
    Ok(())
}

fn append_edge_discretization(
    edge: &SourceTopologyEdge,
    left: [f64; 3],
    right: [f64; 3],
    options: CurveDiscretizationOptions,
    nodes: &mut Vec<CurveNode>,
    elements: &mut Vec<CurveElement>,
) {
    let segment_count = ((edge.length_m / options.target_size_m).ceil() as usize)
        .max(options.min_segments_per_edge)
        .min(options.max_segments_per_edge);
    let first_node_id = nodes.len() as u32;
    for segment_index in 0..=segment_count {
        let parameter = segment_index as f64 / segment_count as f64;
        nodes.push(CurveNode {
            node_id: first_node_id + segment_index as u32,
            source_edge_id: edge.edge_id,
            parameter,
            coordinates_m: interpolate(left, right, parameter),
        });
    }
    for segment_index in 0..segment_count {
        let left_node_id = first_node_id + segment_index as u32;
        let right_node_id = left_node_id + 1;
        let left = nodes[left_node_id as usize].coordinates_m;
        let right = nodes[right_node_id as usize].coordinates_m;
        elements.push(CurveElement {
            element_id: elements.len() as u32,
            source_edge_id: edge.edge_id,
            node_ids: [left_node_id, right_node_id],
            length_m: distance(left, right),
        });
    }
}

fn interpolate(left: [f64; 3], right: [f64; 3], parameter: f64) -> [f64; 3] {
    [
        left[0] + (right[0] - left[0]) * parameter,
        left[1] + (right[1] - left[1]) * parameter,
        left[2] + (right[2] - left[2]) * parameter,
    ]
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_cad::{SourceTopologyEdge, SourceTopologyModel, SourceTopologyVertex};

    #[test]
    fn discretizes_topology_edges_by_target_size() {
        let topology = line_topology(1.0);
        let curves = discretize_topology_curves(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 0.24,
                min_segments_per_edge: 1,
                max_segments_per_edge: 16,
            },
        )
        .expect("curves should discretize");

        assert_eq!(curves.nodes.len(), 6);
        assert_eq!(curves.elements.len(), 5);
        assert_eq!(curves.nodes[0].parameter, 0.0);
        assert_eq!(curves.nodes[5].parameter, 1.0);
        assert!(curves
            .elements
            .iter()
            .all(|element| element.length_m <= 0.200000000001));
    }

    #[test]
    fn discretization_respects_max_segments() {
        let topology = line_topology(1.0);
        let curves = discretize_topology_curves(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 0.01,
                min_segments_per_edge: 1,
                max_segments_per_edge: 4,
            },
        )
        .expect("curves should discretize");

        assert_eq!(curves.elements.len(), 4);
    }

    #[test]
    fn rejects_invalid_curve_options() {
        let err = discretize_topology_curves(
            &line_topology(1.0),
            CurveDiscretizationOptions {
                target_size_m: 0.0,
                ..CurveDiscretizationOptions::default()
            },
        )
        .expect_err("zero target size should fail");

        assert_eq!(err, CurveDiscretizationError::InvalidTargetSize);
    }

    fn line_topology(length_m: f64) -> SourceTopologyModel {
        SourceTopologyModel {
            mesh_id: "line".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
            vertices: vec![
                SourceTopologyVertex {
                    vertex_id: 0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                SourceTopologyVertex {
                    vertex_id: 1,
                    coordinates_m: [length_m, 0.0, 0.0],
                },
            ],
            edges: vec![SourceTopologyEdge {
                edge_id: 0,
                node_ids: [0, 1],
                adjacent_face_ids: Vec::new(),
                region_ids: Vec::new(),
                length_m,
            }],
            faces: Vec::new(),
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [length_m, 0.0, 0.0],
            region_ids: Vec::new(),
        }
    }
}
