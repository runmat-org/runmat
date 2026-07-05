use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use runmat_meshing_cad::{CadTopologyModel, SourceTopologyEdge, SourceTopologyModel};
use runmat_meshing_size::field::{MeshSizingField, SegmentSizingQuery, SizingFieldService};

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadCurveEdgeProvenance {
    pub source_edge_id: u32,
    pub cad_edge_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_curve_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_id: Option<String>,
    #[serde(default)]
    pub evaluator_supports_point_evaluation: bool,
    #[serde(default)]
    pub evaluator_supports_projection: bool,
    #[serde(default)]
    pub evaluator_supports_tangent: bool,
    #[serde(default)]
    pub evaluator_supports_curvature: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadCurveDiscretization {
    pub curves: CurveDiscretization,
    #[serde(default)]
    pub edge_provenance: Vec<CadCurveEdgeProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveDiscretizationError {
    InvalidTargetSize,
    InvalidSegmentBounds,
    MissingEdgeVertex { edge_id: u32, node_id: u32 },
    MissingCadEdge { source_edge_id: u32 },
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
            Self::MissingCadEdge { source_edge_id } => {
                write!(
                    formatter,
                    "CAD topology is missing source edge {source_edge_id}"
                )
            }
        }
    }
}

impl std::error::Error for CurveDiscretizationError {}

pub fn discretize_topology_curves(
    topology: &SourceTopologyModel,
    options: CurveDiscretizationOptions,
) -> Result<CurveDiscretization, CurveDiscretizationError> {
    discretize_topology_curves_with_sizing(topology, options, None)
}

pub fn discretize_topology_curves_with_sizing(
    topology: &SourceTopologyModel,
    options: CurveDiscretizationOptions,
    sizing: Option<&MeshSizingField>,
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
        let target_size_m = sizing
            .and_then(|sizing| {
                sizing
                    .query_segment_size(SegmentSizingQuery {
                        start_m: left,
                        end_m: right,
                    })
                    .target_size_m
            })
            .unwrap_or(options.target_size_m);
        append_edge_discretization(
            edge,
            left,
            right,
            options,
            target_size_m,
            &mut nodes,
            &mut elements,
        );
    }

    Ok(CurveDiscretization { nodes, elements })
}

pub fn discretize_cad_topology_curves_with_sizing(
    topology: &SourceTopologyModel,
    cad_topology: &CadTopologyModel,
    options: CurveDiscretizationOptions,
    sizing: Option<&MeshSizingField>,
) -> Result<CadCurveDiscretization, CurveDiscretizationError> {
    let curves = discretize_topology_curves_with_sizing(topology, options, sizing)?;
    let cad_edges_by_source_edge = cad_topology
        .edges
        .iter()
        .map(|edge| (edge.source_edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let mut edge_provenance = Vec::<CadCurveEdgeProvenance>::with_capacity(topology.edges.len());
    for edge in &topology.edges {
        let cad_edge = cad_edges_by_source_edge.get(&edge.edge_id).ok_or(
            CurveDiscretizationError::MissingCadEdge {
                source_edge_id: edge.edge_id,
            },
        )?;
        edge_provenance.push(CadCurveEdgeProvenance {
            source_edge_id: edge.edge_id,
            cad_edge_id: cad_edge.entity_id.id.clone(),
            imported_curve_id: cad_edge.imported_curve_id,
            evaluator_id: cad_edge.evaluator_id.clone(),
            evaluator_supports_point_evaluation: cad_edge.evaluator_supports_point_evaluation,
            evaluator_supports_projection: cad_edge.evaluator_supports_projection,
            evaluator_supports_tangent: cad_edge.evaluator_supports_tangent,
            evaluator_supports_curvature: cad_edge.evaluator_supports_curvature,
        });
    }
    Ok(CadCurveDiscretization {
        curves,
        edge_provenance,
    })
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
    target_size_m: f64,
    nodes: &mut Vec<CurveNode>,
    elements: &mut Vec<CurveElement>,
) {
    let segment_count = ((edge.length_m / target_size_m).ceil() as usize)
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
    use runmat_meshing_cad::{
        CadEdge, CadEntityId, CadEntityKind, CadTopologyModel, CadTopologyReport,
        CadTopologySource, SourceTopologyEdge, SourceTopologyModel, SourceTopologyVertex,
    };
    use runmat_meshing_size::field::{MeshSizingField, SizingSample};

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
    fn sizing_field_refines_recovered_curve_edges_before_surface_meshing() {
        let topology = line_topology(1.0);
        let sizing = MeshSizingField {
            global_target_size_m: Some(1.0),
            samples: vec![SizingSample {
                position_m: [0.5, 0.0, 0.0],
                target_size_m: 0.2,
                reason: Some("cad.feature_edge".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let curves = discretize_topology_curves_with_sizing(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 1.0,
                min_segments_per_edge: 1,
                max_segments_per_edge: 16,
            },
            Some(&sizing),
        )
        .expect("sizing-aware curves should discretize");

        assert_eq!(curves.nodes.len(), 6);
        assert_eq!(curves.elements.len(), 5);
        assert!(curves
            .elements
            .iter()
            .all(|element| { element.source_edge_id == 0 && element.length_m <= 0.200000000001 }));
    }

    #[test]
    fn sizing_samples_off_curve_do_not_refine_curve_edges() {
        let topology = line_topology(1.0);
        let sizing = MeshSizingField {
            samples: vec![SizingSample {
                position_m: [0.5, 0.5, 0.0],
                target_size_m: 0.1,
                reason: Some("surface_only".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let curves = discretize_topology_curves_with_sizing(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 1.0,
                min_segments_per_edge: 1,
                max_segments_per_edge: 16,
            },
            Some(&sizing),
        )
        .expect("sizing-aware curves should discretize");

        assert_eq!(curves.nodes.len(), 2);
        assert_eq!(curves.elements.len(), 1);
    }

    #[test]
    fn cad_curve_discretization_carries_edge_evaluator_provenance() {
        let topology = line_topology(1.0);
        let cad_topology = cad_topology_for_line(&topology);

        let cad_curves = discretize_cad_topology_curves_with_sizing(
            &topology,
            &cad_topology,
            CurveDiscretizationOptions {
                target_size_m: 0.5,
                min_segments_per_edge: 1,
                max_segments_per_edge: 16,
            },
            None,
        )
        .expect("cad-aware curves should discretize");

        assert_eq!(cad_curves.curves.elements.len(), 2);
        assert_eq!(cad_curves.edge_provenance.len(), 1);
        let provenance = &cad_curves.edge_provenance[0];
        assert_eq!(provenance.source_edge_id, 0);
        assert_eq!(provenance.cad_edge_id, "cad_edge_0");
        assert_eq!(provenance.imported_curve_id, Some(12));
        assert_eq!(provenance.evaluator_id.as_deref(), Some("cad_curve_12"));
        assert!(provenance.evaluator_supports_point_evaluation);
        assert!(provenance.evaluator_supports_projection);
        assert!(provenance.evaluator_supports_tangent);
        assert!(provenance.evaluator_supports_curvature);
    }

    #[test]
    fn cad_curve_discretization_requires_matching_cad_edge() {
        let topology = line_topology(1.0);
        let mut cad_topology = cad_topology_for_line(&topology);
        cad_topology.edges.clear();

        let err = discretize_cad_topology_curves_with_sizing(
            &topology,
            &cad_topology,
            CurveDiscretizationOptions::default(),
            None,
        )
        .expect_err("missing CAD edge should fail");

        assert_eq!(
            err,
            CurveDiscretizationError::MissingCadEdge { source_edge_id: 0 }
        );
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

    fn cad_topology_for_line(topology: &SourceTopologyModel) -> CadTopologyModel {
        CadTopologyModel {
            source_geometry_id: topology.source_geometry_id.clone(),
            source_geometry_revision: topology.source_geometry_revision,
            source_geometry_sha256: topology.source_geometry_sha256.clone(),
            source: CadTopologySource::SemanticCad,
            vertices: Vec::new(),
            edges: vec![CadEdge {
                entity_id: CadEntityId {
                    kind: CadEntityKind::Edge,
                    id: "cad_edge_0".to_string(),
                },
                source_edge_id: 0,
                imported_curve_id: Some(12),
                evaluator_id: Some("cad_curve_12".to_string()),
                evaluator_supports_point_evaluation: true,
                evaluator_supports_projection: true,
                evaluator_supports_tangent: true,
                evaluator_supports_curvature: true,
                vertex_ids: ["cad_vertex_0".to_string(), "cad_vertex_1".to_string()],
                adjacent_face_ids: Vec::new(),
                length_m: 1.0,
            }],
            loops: Vec::new(),
            faces: Vec::new(),
            shells: Vec::new(),
            volumes: Vec::new(),
            report: CadTopologyReport {
                source: CadTopologySource::SemanticCad,
                vertex_count: 0,
                edge_count: 1,
                face_count: 0,
                shell_count: 0,
                volume_count: 0,
                semantic_face_count: 0,
                imported_face_count: 0,
                evaluator_face_count: 0,
                imported_curve_count: 1,
                evaluator_curve_count: 1,
                generic_face_count: 0,
                loop_count: 0,
                hole_loop_count: 0,
                closed_shell_count: 0,
            },
        }
    }
}
