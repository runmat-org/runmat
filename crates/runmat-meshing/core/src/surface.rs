use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    cad_eval::{project_to_face, CadEvaluationModel},
    curve::{CurveDiscretization, CurveNode},
    predicate::triangle_area,
    source_topology::{SourceTopologyFace, SourceTopologyModel},
};

pub const INTERNAL_SOURCE_EDGE_ID: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceDiscretizationOptions {
    pub preserve_source_faces: bool,
    pub centroid_subdivision: bool,
    pub max_curve_segments_per_edge: usize,
}

impl Default for SurfaceDiscretizationOptions {
    fn default() -> Self {
        Self {
            preserve_source_faces: true,
            centroid_subdivision: false,
            max_curve_segments_per_edge: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceNode {
    pub node_id: u32,
    pub source_vertex_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceElement {
    pub element_id: u32,
    pub source_face_id: u32,
    #[serde(default)]
    pub cad_face_id: Option<String>,
    pub source_edge_ids: [u32; 3],
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub parametric_node_uv: [[f64; 2]; 3],
    #[serde(default)]
    pub max_projection_error_m: f64,
    pub region_ids: Vec<String>,
    pub area_m2: f64,
    pub unit_normal: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceDiscretization {
    pub nodes: Vec<SurfaceNode>,
    pub elements: Vec<SurfaceElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceDiscretizationError {
    MissingFaceVertex { face_id: u32, node_id: u32 },
    MissingFaceEdge { face_id: u32, edge_id: u32 },
    MissingCadFaceFrame { source_face_id: u32 },
    MissingCurveEdge { source_edge_id: u32 },
    InvalidFaceEdgeOrientation { face_id: u32, edge_id: u32 },
}

impl std::fmt::Display for SurfaceDiscretizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFaceVertex { face_id, node_id } => write!(
                formatter,
                "source face {face_id} references missing topology vertex {node_id}"
            ),
            Self::MissingFaceEdge { face_id, edge_id } => write!(
                formatter,
                "source face {face_id} references missing topology edge {edge_id}"
            ),
            Self::MissingCadFaceFrame { source_face_id } => write!(
                formatter,
                "source face {source_face_id} does not have a CAD evaluation frame"
            ),
            Self::MissingCurveEdge { source_edge_id } => write!(
                formatter,
                "source edge {source_edge_id} does not have curve discretization nodes"
            ),
            Self::InvalidFaceEdgeOrientation { face_id, edge_id } => write!(
                formatter,
                "source face {face_id} cannot orient source edge {edge_id} along its boundary"
            ),
        }
    }
}

impl std::error::Error for SurfaceDiscretizationError {}

pub fn discretize_topology_surfaces(
    topology: &SourceTopologyModel,
    _options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();

    let mut elements = Vec::<SurfaceElement>::with_capacity(topology.faces.len());
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: None,
            source_edge_ids: face.edge_ids,
            node_ids: face.node_ids,
            parametric_node_uv: [[0.0, 0.0]; 3],
            max_projection_error_m: 0.0,
            region_ids: face.region_ids.clone(),
            area_m2: face.area_m2,
            unit_normal: face.unit_normal,
        });
    }

    Ok(SurfaceDiscretization { nodes, elements })
}

pub fn discretize_cad_surfaces(
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let mut nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();
    let frames_by_source_face = cad_evaluation
        .face_frames
        .iter()
        .map(|frame| (frame.source_face_id, frame))
        .collect::<BTreeMap<_, _>>();

    let element_capacity = if options.centroid_subdivision {
        topology.faces.len() * 3
    } else {
        topology.faces.len()
    };
    let mut elements = Vec::<SurfaceElement>::with_capacity(element_capacity);
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        let frame = frames_by_source_face.get(&face.face_id).ok_or(
            SurfaceDiscretizationError::MissingCadFaceFrame {
                source_face_id: face.face_id,
            },
        )?;
        let mut parametric_node_uv = [[0.0_f64, 0.0_f64]; 3];
        let mut max_projection_error_m = 0.0_f64;
        let mut corner_points = [[0.0_f64, 0.0_f64, 0.0_f64]; 3];
        for (index, node_id) in face.node_ids.into_iter().enumerate() {
            let point = topology
                .vertices
                .get(node_id as usize)
                .filter(|vertex| vertex.vertex_id == node_id)
                .map(|vertex| vertex.coordinates_m)
                .ok_or(SurfaceDiscretizationError::MissingFaceVertex {
                    face_id: face.face_id,
                    node_id,
                })?;
            corner_points[index] = point;
            let projection = project_to_face(frame, point);
            parametric_node_uv[index] = projection.uv;
            max_projection_error_m = max_projection_error_m.max(projection.distance_m);
        }

        if options.centroid_subdivision {
            let centroid = triangle_centroid(corner_points);
            let centroid_projection = project_to_face(frame, centroid);
            max_projection_error_m = max_projection_error_m.max(centroid_projection.distance_m);
            append_centroid_subdivision(
                face,
                frame,
                parametric_node_uv,
                centroid,
                centroid_projection.uv,
                max_projection_error_m,
                &mut nodes,
                &mut elements,
            );
        } else {
            elements.push(SurfaceElement {
                element_id: elements.len() as u32,
                source_face_id: face.face_id,
                cad_face_id: Some(frame.face_id.clone()),
                source_edge_ids: face.edge_ids,
                node_ids: face.node_ids,
                parametric_node_uv,
                max_projection_error_m,
                region_ids: face.region_ids.clone(),
                area_m2: face.area_m2,
                unit_normal: frame.unit_normal,
            });
        }
    }

    Ok(SurfaceDiscretization { nodes, elements })
}

pub fn discretize_cad_surfaces_with_curves(
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    curves: &CurveDiscretization,
    options: SurfaceDiscretizationOptions,
) -> Result<SurfaceDiscretization, SurfaceDiscretizationError> {
    let mut nodes = topology
        .vertices
        .iter()
        .map(|vertex| SurfaceNode {
            node_id: vertex.vertex_id,
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();
    let frames_by_source_face = cad_evaluation
        .face_frames
        .iter()
        .map(|frame| (frame.source_face_id, frame))
        .collect::<BTreeMap<_, _>>();
    let topology_edges = topology
        .edges
        .iter()
        .map(|edge| (edge.edge_id, edge))
        .collect::<BTreeMap<_, _>>();
    let curve_nodes_by_edge = curve_nodes_by_source_edge(curves);
    let mut curve_node_to_surface_node = BTreeMap::<u32, u32>::new();

    let mut elements = Vec::<SurfaceElement>::new();
    for face in &topology.faces {
        validate_face_vertices(topology, face)?;
        let frame = frames_by_source_face.get(&face.face_id).ok_or(
            SurfaceDiscretizationError::MissingCadFaceFrame {
                source_face_id: face.face_id,
            },
        )?;
        let segments = oriented_face_curve_segments(
            &topology_edges,
            &curve_nodes_by_edge,
            face,
            options.max_curve_segments_per_edge.max(1),
            &mut nodes,
            &mut curve_node_to_surface_node,
        )?;
        let centroid = face_centroid_from_segments(&nodes, &segments);
        let centroid_projection = project_to_face(frame, centroid);
        let centroid_node_id = nodes.len() as u32;
        nodes.push(SurfaceNode {
            node_id: centroid_node_id,
            source_vertex_id: u32::MAX,
            coordinates_m: centroid,
        });

        for segment in segments {
            let points = [
                nodes[segment.node_ids[0] as usize].coordinates_m,
                nodes[segment.node_ids[1] as usize].coordinates_m,
                centroid,
            ];
            let left_projection = project_to_face(frame, points[0]);
            let right_projection = project_to_face(frame, points[1]);
            let max_projection_error_m = left_projection
                .distance_m
                .max(right_projection.distance_m)
                .max(centroid_projection.distance_m);
            elements.push(SurfaceElement {
                element_id: elements.len() as u32,
                source_face_id: face.face_id,
                cad_face_id: Some(frame.face_id.clone()),
                source_edge_ids: [
                    segment.source_edge_id,
                    INTERNAL_SOURCE_EDGE_ID,
                    INTERNAL_SOURCE_EDGE_ID,
                ],
                node_ids: [segment.node_ids[0], segment.node_ids[1], centroid_node_id],
                parametric_node_uv: [
                    left_projection.uv,
                    right_projection.uv,
                    centroid_projection.uv,
                ],
                max_projection_error_m,
                region_ids: face.region_ids.clone(),
                area_m2: triangle_area(points),
                unit_normal: frame.unit_normal,
            });
        }
    }

    Ok(SurfaceDiscretization { nodes, elements })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaceCurveSegment {
    node_ids: [u32; 2],
    source_edge_id: u32,
}

fn curve_nodes_by_source_edge(curves: &CurveDiscretization) -> BTreeMap<u32, Vec<&CurveNode>> {
    let mut by_edge = BTreeMap::<u32, Vec<&CurveNode>>::new();
    for node in &curves.nodes {
        by_edge.entry(node.source_edge_id).or_default().push(node);
    }
    for nodes in by_edge.values_mut() {
        nodes.sort_by(|left, right| left.parameter.total_cmp(&right.parameter));
    }
    by_edge
}

fn oriented_face_curve_segments(
    topology_edges: &BTreeMap<u32, &crate::SourceTopologyEdge>,
    curve_nodes_by_edge: &BTreeMap<u32, Vec<&CurveNode>>,
    face: &SourceTopologyFace,
    max_curve_segments_per_edge: usize,
    nodes: &mut Vec<SurfaceNode>,
    curve_node_to_surface_node: &mut BTreeMap<u32, u32>,
) -> Result<Vec<FaceCurveSegment>, SurfaceDiscretizationError> {
    let mut segments = Vec::<FaceCurveSegment>::new();
    for edge_index in 0..3 {
        let desired_start = face.node_ids[edge_index];
        let desired_end = face.node_ids[(edge_index + 1) % 3];
        let (source_edge_id, edge) =
            face_edge_for_nodes(face, topology_edges, desired_start, desired_end)?;
        let curve_nodes = curve_nodes_by_edge
            .get(&source_edge_id)
            .ok_or(SurfaceDiscretizationError::MissingCurveEdge { source_edge_id })?;
        let reverse = if edge.node_ids == [desired_start, desired_end] {
            false
        } else if edge.node_ids == [desired_end, desired_start] {
            true
        } else {
            return Err(SurfaceDiscretizationError::InvalidFaceEdgeOrientation {
                face_id: face.face_id,
                edge_id: source_edge_id,
            });
        };
        let capped_nodes = capped_curve_nodes(curve_nodes.clone(), max_curve_segments_per_edge);
        let ordered_nodes = if reverse {
            capped_nodes.iter().rev().copied().collect::<Vec<_>>()
        } else {
            capped_nodes
        };
        for pair in ordered_nodes.windows(2) {
            let left = surface_node_for_curve_node(
                edge.node_ids,
                pair[0],
                nodes,
                curve_node_to_surface_node,
            )?;
            let right = surface_node_for_curve_node(
                edge.node_ids,
                pair[1],
                nodes,
                curve_node_to_surface_node,
            )?;
            if left != right {
                segments.push(FaceCurveSegment {
                    node_ids: [left, right],
                    source_edge_id,
                });
            }
        }
    }
    Ok(segments)
}

fn face_edge_for_nodes<'a>(
    face: &SourceTopologyFace,
    topology_edges: &'a BTreeMap<u32, &crate::SourceTopologyEdge>,
    desired_start: u32,
    desired_end: u32,
) -> Result<(u32, &'a crate::SourceTopologyEdge), SurfaceDiscretizationError> {
    for source_edge_id in face.edge_ids {
        let edge = topology_edges.get(&source_edge_id).ok_or(
            SurfaceDiscretizationError::MissingFaceEdge {
                face_id: face.face_id,
                edge_id: source_edge_id,
            },
        )?;
        if edge.node_ids == [desired_start, desired_end]
            || edge.node_ids == [desired_end, desired_start]
        {
            return Ok((source_edge_id, edge));
        }
    }
    Err(SurfaceDiscretizationError::InvalidFaceEdgeOrientation {
        face_id: face.face_id,
        edge_id: face.edge_ids[0],
    })
}

fn surface_node_for_curve_node(
    edge_node_ids: [u32; 2],
    curve_node: &CurveNode,
    nodes: &mut Vec<SurfaceNode>,
    curve_node_to_surface_node: &mut BTreeMap<u32, u32>,
) -> Result<u32, SurfaceDiscretizationError> {
    if curve_node.parameter <= f64::EPSILON {
        return Ok(edge_node_ids[0]);
    }
    if (1.0 - curve_node.parameter).abs() <= f64::EPSILON {
        return Ok(edge_node_ids[1]);
    }
    if let Some(node_id) = curve_node_to_surface_node.get(&curve_node.node_id) {
        return Ok(*node_id);
    }
    let node_id = nodes.len() as u32;
    curve_node_to_surface_node.insert(curve_node.node_id, node_id);
    nodes.push(SurfaceNode {
        node_id,
        source_vertex_id: u32::MAX,
        coordinates_m: curve_node.coordinates_m,
    });
    Ok(node_id)
}

fn capped_curve_nodes<'a>(
    curve_nodes: Vec<&'a CurveNode>,
    max_segments_per_edge: usize,
) -> Vec<&'a CurveNode> {
    if curve_nodes.len() <= max_segments_per_edge.saturating_add(1) {
        return curve_nodes;
    }
    let segment_count = max_segments_per_edge.max(1);
    let last_index = curve_nodes.len() - 1;
    let mut capped = Vec::<&CurveNode>::with_capacity(segment_count + 1);
    for index in 0..=segment_count {
        let source_index = ((index * last_index) + (segment_count / 2)) / segment_count;
        let source_index = source_index.min(last_index);
        if capped
            .last()
            .is_none_or(|node| node.node_id != curve_nodes[source_index].node_id)
        {
            capped.push(curve_nodes[source_index]);
        }
    }
    capped
}

fn face_centroid_from_segments(nodes: &[SurfaceNode], segments: &[FaceCurveSegment]) -> [f64; 3] {
    let mut sum = [0.0_f64; 3];
    let mut count = 0.0_f64;
    for segment in segments {
        let point = nodes[segment.node_ids[0] as usize].coordinates_m;
        sum[0] += point[0];
        sum[1] += point[1];
        sum[2] += point[2];
        count += 1.0;
    }
    if count <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [sum[0] / count, sum[1] / count, sum[2] / count]
}

fn append_centroid_subdivision(
    face: &SourceTopologyFace,
    frame: &crate::CadFaceEvaluationFrame,
    corner_uv: [[f64; 2]; 3],
    centroid_m: [f64; 3],
    centroid_uv: [f64; 2],
    corner_projection_error_m: f64,
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) {
    let centroid_node_id = nodes.len() as u32;
    nodes.push(SurfaceNode {
        node_id: centroid_node_id,
        source_vertex_id: u32::MAX,
        coordinates_m: centroid_m,
    });
    let child_specs = [
        (
            [face.node_ids[0], face.node_ids[1], centroid_node_id],
            [face.edge_ids[0], face.edge_ids[1], face.edge_ids[2]],
            [corner_uv[0], corner_uv[1], centroid_uv],
        ),
        (
            [face.node_ids[1], face.node_ids[2], centroid_node_id],
            [face.edge_ids[1], face.edge_ids[2], face.edge_ids[0]],
            [corner_uv[1], corner_uv[2], centroid_uv],
        ),
        (
            [face.node_ids[2], face.node_ids[0], centroid_node_id],
            [face.edge_ids[2], face.edge_ids[0], face.edge_ids[1]],
            [corner_uv[2], corner_uv[0], centroid_uv],
        ),
    ];
    for (node_ids, source_edge_ids, parametric_node_uv) in child_specs {
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: Some(frame.face_id.clone()),
            source_edge_ids,
            node_ids,
            parametric_node_uv,
            max_projection_error_m: corner_projection_error_m,
            region_ids: face.region_ids.clone(),
            area_m2: face.area_m2 / 3.0,
            unit_normal: frame.unit_normal,
        });
    }
}

fn triangle_centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
        (points[0][2] + points[1][2] + points[2][2]) / 3.0,
    ]
}

fn validate_face_vertices(
    topology: &SourceTopologyModel,
    face: &SourceTopologyFace,
) -> Result<(), SurfaceDiscretizationError> {
    for node_id in face.node_ids {
        if topology
            .vertices
            .get(node_id as usize)
            .is_none_or(|vertex| vertex.vertex_id != node_id)
        {
            return Err(SurfaceDiscretizationError::MissingFaceVertex {
                face_id: face.face_id,
                node_id,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        SourceTopologyEdge, SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex,
    };

    #[test]
    fn discretizes_source_faces_as_surface_elements() {
        let surface = discretize_topology_surfaces(
            &single_triangle_topology(),
            SurfaceDiscretizationOptions::default(),
        )
        .expect("surface should discretize");

        assert_eq!(surface.nodes.len(), 3);
        assert_eq!(surface.elements.len(), 1);
        assert_eq!(surface.elements[0].source_face_id, 7);
        assert_eq!(surface.elements[0].cad_face_id, None);
        assert_eq!(surface.elements[0].source_edge_ids, [0, 1, 2]);
        assert_eq!(surface.elements[0].parametric_node_uv, [[0.0, 0.0]; 3]);
        assert_eq!(surface.elements[0].max_projection_error_m, 0.0);
        assert_eq!(surface.elements[0].region_ids, vec!["face_a".to_string()]);
        assert!((surface.elements[0].area_m2 - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn discretizes_surfaces_with_cad_face_ownership() {
        let topology = single_triangle_topology();
        let cad_topology =
            crate::build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

        let surface = discretize_cad_surfaces(
            &topology,
            &cad_evaluation,
            SurfaceDiscretizationOptions::default(),
        )
        .expect("cad-owned surface should discretize");

        assert_eq!(surface.elements.len(), 1);
        assert_eq!(
            surface.elements[0].cad_face_id,
            Some("cad_face_7".to_string())
        );
        assert_eq!(surface.elements[0].parametric_node_uv.len(), 3);
        assert_eq!(surface.elements[0].max_projection_error_m, 0.0);
    }

    #[test]
    fn centroid_subdivision_preserves_cad_face_ownership_and_boundary_edges() {
        let topology = single_triangle_topology();
        let cad_topology =
            crate::build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

        let surface = discretize_cad_surfaces(
            &topology,
            &cad_evaluation,
            SurfaceDiscretizationOptions {
                centroid_subdivision: true,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("cad-owned surface should subdivide");

        assert_eq!(surface.nodes.len(), 4);
        assert_eq!(surface.elements.len(), 3);
        assert!(surface
            .elements
            .iter()
            .all(|element| element.cad_face_id == Some("cad_face_7".to_string())));
        assert!(surface.elements.iter().any(|element| {
            element.node_ids[0..2] == [0, 1] && element.source_edge_ids[0] == 0
        }));
        assert_eq!(surface.nodes[3].coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    }

    #[test]
    fn curve_driven_cad_surface_uses_curve_boundary_nodes() {
        let topology = single_triangle_topology();
        let cad_topology =
            crate::build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = crate::discretize_topology_curves(
            &topology,
            crate::CurveDiscretizationOptions {
                target_size_m: 0.25,
                min_segments_per_edge: 2,
                max_segments_per_edge: 2,
            },
        )
        .expect("curves should discretize");

        let surface = discretize_cad_surfaces_with_curves(
            &topology,
            &cad_evaluation,
            &curves,
            SurfaceDiscretizationOptions {
                max_curve_segments_per_edge: 2,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("cad-owned curve surface should discretize");

        assert_eq!(surface.elements.len(), 6);
        assert!(surface.nodes.len() > topology.vertices.len());
        assert!(surface.elements.iter().all(|element| {
            element.cad_face_id == Some("cad_face_7".to_string())
                && element.source_edge_ids[1] == INTERNAL_SOURCE_EDGE_ID
                && element.source_edge_ids[2] == INTERNAL_SOURCE_EDGE_ID
        }));
    }

    #[test]
    fn rejects_missing_face_vertices() {
        let mut topology = single_triangle_topology();
        topology.vertices.pop();

        let err = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
            .expect_err("missing face vertex should fail");

        assert_eq!(
            err,
            SurfaceDiscretizationError::MissingFaceVertex {
                face_id: 7,
                node_id: 2,
            }
        );
    }

    fn single_triangle_topology() -> SourceTopologyModel {
        SourceTopologyModel {
            mesh_id: "surface".to_string(),
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
                    coordinates_m: [1.0, 0.0, 0.0],
                },
                SourceTopologyVertex {
                    vertex_id: 2,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
            ],
            edges: vec![
                SourceTopologyEdge {
                    edge_id: 0,
                    node_ids: [0, 1],
                    adjacent_face_ids: vec![7],
                    region_ids: vec!["face_a".to_string()],
                    length_m: 1.0,
                },
                SourceTopologyEdge {
                    edge_id: 1,
                    node_ids: [1, 2],
                    adjacent_face_ids: vec![7],
                    region_ids: vec!["face_a".to_string()],
                    length_m: 2.0_f64.sqrt(),
                },
                SourceTopologyEdge {
                    edge_id: 2,
                    node_ids: [0, 2],
                    adjacent_face_ids: vec![7],
                    region_ids: vec!["face_a".to_string()],
                    length_m: 1.0,
                },
            ],
            faces: vec![SourceTopologyFace {
                face_id: 7,
                source_triangle_id: 11,
                node_ids: [0, 1, 2],
                edge_ids: [0, 1, 2],
                region_ids: vec!["face_a".to_string()],
                area_m2: 0.5,
                unit_normal: [0.0, 0.0, 1.0],
            }],
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 0.0],
            region_ids: vec!["face_a".to_string()],
        }
    }

    fn geometry_for_topology() -> runmat_geometry_core::GeometryAsset {
        runmat_geometry_core::GeometryAsset {
            geometry_id: "geo".to_string(),
            source: runmat_geometry_core::GeometrySource {
                path: "/fixtures/surface.step".to_string(),
                sha256: "surface".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: runmat_geometry_core::SourceGeometry {
                kind: runmat_geometry_core::SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: runmat_geometry_core::TessellationProfile::default(),
            units: runmat_geometry_core::UnitSystem::Meter,
            revision: 1,
            meshes: Vec::new(),
            surface_meshes: Vec::new(),
            regions: Vec::new(),
            region_entity_mappings: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}
