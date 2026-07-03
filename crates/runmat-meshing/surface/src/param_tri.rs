use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::math::{cross, dot, sub, triangle_area};
use geometry::{boundary_loop_polygons, sorted_node_pair, triangle_edges_2d};
use runmat_meshing_cad::{
    project_to_face, CadEvaluationModel, CadFaceEvaluationFrame, SourceTopologyEdge,
    SourceTopologyFace, SourceTopologyModel,
};
use runmat_meshing_curve::{CurveDiscretization, CurveNode};

mod geometry;
mod sampling;
mod triangulation;

use sampling::{
    append_exact_face_domain_sample_points, append_face_lattice_points, face_area_is_recovered,
    face_edges_are_recovered, has_exact_face_domain_samples, ExactCadSampleSurfaceReport,
};
use triangulation::{triangulate_face_points, triangulate_triangle_points_by_insertion};

#[cfg(test)]
use geometry::{point_in_polygon_2d, triangle_centroid_2d};

pub const MODULE_PURPOSE: &str = "face-domain triangulation from recovered curve boundaries";
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
    #[serde(default)]
    pub exact_cad_sample_node_count: usize,
    #[serde(default)]
    pub rejected_exact_cad_sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceDiscretizationError {
    MissingFaceVertex {
        face_id: u32,
        node_id: u32,
    },
    MissingFaceEdge {
        face_id: u32,
        edge_id: u32,
    },
    MissingCadFaceFrame {
        source_face_id: u32,
    },
    MissingCurveEdge {
        source_edge_id: u32,
    },
    InvalidFaceEdgeOrientation {
        face_id: u32,
        edge_id: u32,
    },
    CadProjectionOutsideFaceDomain {
        face_id: u32,
        node_id: u32,
    },
    EmptyFaceLoop {
        face_id: u32,
    },
    InvalidFaceLoopTopology {
        face_id: u32,
        node_id: u32,
        incident_segment_count: usize,
    },
    MultipleFaceLoopsUnsupported {
        face_id: u32,
        loop_count: usize,
        loop_node_counts: Vec<usize>,
        loop_source_edge_ids: Vec<Vec<u32>>,
    },
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
            Self::CadProjectionOutsideFaceDomain { face_id, node_id } => write!(
                formatter,
                "source face {face_id} node {node_id} projects outside the CAD face domain"
            ),
            Self::EmptyFaceLoop { face_id } => {
                write!(formatter, "source face {face_id} has an empty boundary loop")
            }
            Self::InvalidFaceLoopTopology {
                face_id,
                node_id,
                incident_segment_count,
            } => write!(
                formatter,
                "source face {face_id} boundary node {node_id} has {incident_segment_count} incident curve segments"
            ),
            Self::MultipleFaceLoopsUnsupported {
                face_id,
                loop_count,
                loop_node_counts,
                loop_source_edge_ids,
            } => write!(
                formatter,
                "source face {face_id} has {loop_count} boundary loops with node counts {loop_node_counts:?} and source edge loops {loop_source_edge_ids:?}; holed or multi-loop faces are not supported by this surface triangulation path yet"
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

    Ok(SurfaceDiscretization {
        nodes,
        elements,
        exact_cad_sample_node_count: 0,
        rejected_exact_cad_sample_count: 0,
    })
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
            if !projection.uv_in_bounds {
                return Err(SurfaceDiscretizationError::CadProjectionOutsideFaceDomain {
                    face_id: face.face_id,
                    node_id,
                });
            }
            parametric_node_uv[index] = projection.uv;
            max_projection_error_m = max_projection_error_m.max(projection.distance_m);
        }

        if options.centroid_subdivision {
            let centroid = triangle_centroid(corner_points);
            let centroid_projection = project_to_face(frame, centroid);
            if !centroid_projection.uv_in_bounds {
                return Err(SurfaceDiscretizationError::CadProjectionOutsideFaceDomain {
                    face_id: face.face_id,
                    node_id: u32::MAX,
                });
            }
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

    Ok(SurfaceDiscretization {
        nodes,
        elements,
        exact_cad_sample_node_count: 0,
        rejected_exact_cad_sample_count: 0,
    })
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
    let mut exact_cad_sample_node_count = 0_usize;
    let mut rejected_exact_cad_sample_count = 0_usize;
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
        let segment_loops = face_curve_segment_loops(face.face_id, &segments)?;
        let sample_report = append_curve_driven_face_elements(
            face,
            frame,
            &segment_loops,
            &mut nodes,
            &mut elements,
        );
        exact_cad_sample_node_count += sample_report.accepted_count;
        rejected_exact_cad_sample_count += sample_report.rejected_count;
    }

    Ok(SurfaceDiscretization {
        nodes,
        elements,
        exact_cad_sample_node_count,
        rejected_exact_cad_sample_count,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaceCurveSegment {
    node_ids: [u32; 2],
    source_edge_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FaceTriangulationPoint {
    node_id: u32,
    uv: [f64; 2],
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
    topology_edges: &BTreeMap<u32, &SourceTopologyEdge>,
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

#[cfg(test)]
fn single_face_curve_segment_loop(
    face_id: u32,
    segments: &[FaceCurveSegment],
) -> Result<Vec<FaceCurveSegment>, SurfaceDiscretizationError> {
    let loops = face_curve_segment_loops(face_id, segments)?;
    if loops.len() != 1 {
        return Err(SurfaceDiscretizationError::MultipleFaceLoopsUnsupported {
            face_id,
            loop_count: loops.len(),
            loop_node_counts: loops
                .iter()
                .map(|loop_segments| loop_segments.len())
                .collect(),
            loop_source_edge_ids: loops
                .iter()
                .map(|loop_segments| {
                    loop_segments
                        .iter()
                        .map(|segment| segment.source_edge_id)
                        .collect()
                })
                .collect(),
        });
    }
    Ok(loops.into_iter().next().unwrap_or_default())
}

fn face_curve_segment_loops(
    face_id: u32,
    segments: &[FaceCurveSegment],
) -> Result<Vec<Vec<FaceCurveSegment>>, SurfaceDiscretizationError> {
    if segments.is_empty() {
        return Err(SurfaceDiscretizationError::EmptyFaceLoop { face_id });
    }

    let mut adjacency = BTreeMap::<u32, Vec<(u32, FaceCurveSegment)>>::new();
    for segment in segments {
        adjacency
            .entry(segment.node_ids[0])
            .or_default()
            .push((segment.node_ids[1], *segment));
        adjacency
            .entry(segment.node_ids[1])
            .or_default()
            .push((segment.node_ids[0], *segment));
    }

    for (node_id, adjacent_nodes) in &adjacency {
        if adjacent_nodes.len() != 2 {
            return Err(SurfaceDiscretizationError::InvalidFaceLoopTopology {
                face_id,
                node_id: *node_id,
                incident_segment_count: adjacent_nodes.len(),
            });
        }
    }

    let mut visited = BTreeSet::<u32>::new();
    let mut loops = Vec::<Vec<FaceCurveSegment>>::new();
    for node_id in adjacency.keys() {
        if !visited.insert(*node_id) {
            continue;
        }
        let start = *node_id;
        let mut current = start;
        let mut previous = None::<u32>;
        let mut loop_segments = Vec::<FaceCurveSegment>::new();
        loop {
            let adjacent_nodes = adjacency.get(&current).ok_or(
                SurfaceDiscretizationError::InvalidFaceLoopTopology {
                    face_id,
                    node_id: current,
                    incident_segment_count: 0,
                },
            )?;
            let next = adjacent_nodes
                .iter()
                .copied()
                .filter(|(neighbor, _)| Some(*neighbor) != previous)
                .min_by_key(|(neighbor, _)| *neighbor)
                .ok_or(SurfaceDiscretizationError::InvalidFaceLoopTopology {
                    face_id,
                    node_id: current,
                    incident_segment_count: adjacent_nodes.len(),
                })?;
            let oriented_segment = FaceCurveSegment {
                node_ids: [current, next.0],
                source_edge_id: next.1.source_edge_id,
            };
            loop_segments.push(oriented_segment);
            previous = Some(current);
            current = next.0;
            if current == start {
                break;
            }
            if !visited.insert(current) {
                return Err(SurfaceDiscretizationError::InvalidFaceLoopTopology {
                    face_id,
                    node_id: current,
                    incident_segment_count: adjacent_nodes.len(),
                });
            }
            if loop_segments.len() > segments.len() {
                return Err(SurfaceDiscretizationError::InvalidFaceLoopTopology {
                    face_id,
                    node_id: current,
                    incident_segment_count: adjacent_nodes.len(),
                });
            }
        }
        if loop_segments.len() < 3 {
            return Err(SurfaceDiscretizationError::InvalidFaceLoopTopology {
                face_id,
                node_id: start,
                incident_segment_count: loop_segments.len(),
            });
        }
        for segment in &loop_segments {
            for node_id in segment.node_ids {
                if !visited.contains(&node_id) {
                    visited.insert(node_id);
                }
            }
        }
        loops.push(loop_segments);
    }

    Ok(loops)
}

fn face_edge_for_nodes<'a>(
    face: &SourceTopologyFace,
    topology_edges: &'a BTreeMap<u32, &SourceTopologyEdge>,
    desired_start: u32,
    desired_end: u32,
) -> Result<(u32, &'a SourceTopologyEdge), SurfaceDiscretizationError> {
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

fn append_curve_driven_face_elements(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    segment_loops: &[Vec<FaceCurveSegment>],
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) -> ExactCadSampleSurfaceReport {
    let segments = segment_loops
        .iter()
        .flat_map(|loop_segments| loop_segments.iter().copied())
        .collect::<Vec<_>>();
    if segments.len() <= 3 && !has_exact_face_domain_samples(frame) {
        append_curve_fan_face_elements(face, frame, &segments, nodes, elements);
        return ExactCadSampleSurfaceReport::default();
    }

    let node_start = nodes.len();
    let element_start = elements.len();
    let mut boundary_edge_ids = BTreeMap::<[u32; 2], u32>::new();
    for segment in &segments {
        boundary_edge_ids.insert(
            sorted_node_pair(segment.node_ids[0], segment.node_ids[1]),
            segment.source_edge_id,
        );
    }

    let mut points = boundary_triangulation_points(frame, &segments, nodes);
    let boundary_point_count = points.len();
    let boundary_polygons = boundary_loop_polygons(frame, segment_loops, nodes);
    let sample_report =
        append_exact_face_domain_sample_points(face, frame, &boundary_polygons, nodes, &mut points);
    append_face_lattice_points(
        face,
        frame,
        &boundary_polygons,
        &segments,
        nodes,
        &mut points,
    );
    let triangles = if segment_loops.len() == 1 && boundary_point_count == 3 {
        triangulate_triangle_points_by_insertion(&points, boundary_point_count)
    } else {
        triangulate_face_points(&points, &boundary_polygons)
    };
    if triangles.is_empty() {
        append_curve_fan_face_elements(face, frame, &segments, nodes, elements);
        return sample_report;
    }

    for triangle in triangles {
        let mut node_ids = triangle.point_indices.map(|index| points[index].node_id);
        if node_ids[0] == node_ids[1] || node_ids[1] == node_ids[2] || node_ids[2] == node_ids[0] {
            continue;
        }
        let mut parametric_node_uv = triangle.point_indices.map(|index| points[index].uv);
        let mut coordinates = node_ids.map(|node_id| nodes[node_id as usize].coordinates_m);
        if triangle_area(coordinates) <= f64::EPSILON {
            continue;
        }
        if dot(
            cross(
                sub(coordinates[1], coordinates[0]),
                sub(coordinates[2], coordinates[0]),
            ),
            frame.unit_normal,
        ) < 0.0
        {
            node_ids.swap(1, 2);
            parametric_node_uv.swap(1, 2);
            coordinates.swap(1, 2);
        }
        let source_edge_ids = [
            boundary_edge_ids
                .get(&sorted_node_pair(node_ids[0], node_ids[1]))
                .copied()
                .unwrap_or(INTERNAL_SOURCE_EDGE_ID),
            boundary_edge_ids
                .get(&sorted_node_pair(node_ids[1], node_ids[2]))
                .copied()
                .unwrap_or(INTERNAL_SOURCE_EDGE_ID),
            boundary_edge_ids
                .get(&sorted_node_pair(node_ids[2], node_ids[0]))
                .copied()
                .unwrap_or(INTERNAL_SOURCE_EDGE_ID),
        ];
        let max_projection_error_m = node_ids
            .iter()
            .map(|node_id| {
                project_to_face(frame, nodes[*node_id as usize].coordinates_m).distance_m
            })
            .fold(0.0_f64, f64::max);
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: Some(frame.face_id.clone()),
            source_edge_ids,
            node_ids,
            parametric_node_uv,
            max_projection_error_m,
            region_ids: face.region_ids.clone(),
            area_m2: triangle_area(coordinates),
            unit_normal: frame.unit_normal,
        });
    }
    if !face_area_is_recovered(face, &elements[element_start..])
        || !face_edges_are_recovered(&elements[element_start..], &boundary_edge_ids)
    {
        nodes.truncate(node_start);
        elements.truncate(element_start);
        append_curve_fan_face_elements(face, frame, &segments, nodes, elements);
        return sample_report.rejected_after_area_guard();
    }
    sample_report
}

fn append_curve_fan_face_elements(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    segments: &[FaceCurveSegment],
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) {
    let centroid = face_centroid_from_segments(nodes, segments);
    let centroid_projection = project_to_face(frame, centroid);
    let centroid_node_id = nodes.len() as u32;
    nodes.push(SurfaceNode {
        node_id: centroid_node_id,
        source_vertex_id: u32::MAX,
        coordinates_m: centroid,
    });

    for segment in segments {
        let mut node_ids = [segment.node_ids[0], segment.node_ids[1], centroid_node_id];
        let mut points = [
            nodes[segment.node_ids[0] as usize].coordinates_m,
            nodes[segment.node_ids[1] as usize].coordinates_m,
            centroid,
        ];
        let mut parametric_node_uv = [
            project_to_face(frame, points[0]).uv,
            project_to_face(frame, points[1]).uv,
            centroid_projection.uv,
        ];
        if dot(
            cross(sub(points[1], points[0]), sub(points[2], points[0])),
            frame.unit_normal,
        ) < 0.0
        {
            node_ids.swap(1, 2);
            points.swap(1, 2);
            parametric_node_uv.swap(1, 2);
        }
        let left_projection = project_to_face(frame, points[0]);
        let right_projection = project_to_face(frame, points[1]);
        let max_projection_error_m = left_projection
            .distance_m
            .max(right_projection.distance_m)
            .max(centroid_projection.distance_m);
        let segment_edge = sorted_node_pair(segment.node_ids[0], segment.node_ids[1]);
        let source_edge_ids = triangle_edges_2d([0, 1, 2]).map(|edge| {
            if sorted_node_pair(node_ids[edge[0]], node_ids[edge[1]]) == segment_edge {
                segment.source_edge_id
            } else {
                INTERNAL_SOURCE_EDGE_ID
            }
        });
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: Some(frame.face_id.clone()),
            source_edge_ids,
            node_ids,
            parametric_node_uv,
            max_projection_error_m,
            region_ids: face.region_ids.clone(),
            area_m2: triangle_area(points),
            unit_normal: frame.unit_normal,
        });
    }
}

fn boundary_triangulation_points(
    frame: &CadFaceEvaluationFrame,
    segments: &[FaceCurveSegment],
    nodes: &[SurfaceNode],
) -> Vec<FaceTriangulationPoint> {
    let mut points = Vec::<FaceTriangulationPoint>::new();
    for segment in segments {
        for node_id in segment.node_ids {
            if points.iter().any(|point| point.node_id == node_id) {
                continue;
            }
            points.push(FaceTriangulationPoint {
                node_id,
                uv: project_to_face(frame, nodes[node_id as usize].coordinates_m).uv,
            });
        }
    }
    points
}

fn append_centroid_subdivision(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
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
    use runmat_geometry_core::{
        CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
        CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, Region,
        RegionEntityMapping,
    };
    use runmat_meshing_cad::{
        build_cad_evaluation_model, build_cad_topology, CadFaceEvaluationFrame, SourceTopologyEdge,
        SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex,
    };
    use runmat_meshing_curve::{
        discretize_topology_curves, CurveDiscretization, CurveDiscretizationOptions, CurveElement,
        CurveNode,
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
            build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

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
    fn rejects_cad_surface_vertex_outside_uv_domain() {
        let topology = single_triangle_topology();
        let mut geometry = geometry_with_face_domain_sample();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.0, 0.0, 0.0],
                uv: Some([0.0, 0.0]),
                projected_point_m: Some([0.0, 0.0, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.0, 0.0],
                uv: Some([0.5, 0.0]),
                projected_point_m: Some([0.5, 0.0, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.0, 0.5, 0.0],
                uv: Some([0.0, 0.5]),
                projected_point_m: Some([0.0, 0.5, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
        ];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

        let err = discretize_cad_surfaces(
            &topology,
            &cad_evaluation,
            SurfaceDiscretizationOptions::default(),
        )
        .expect_err("out-of-domain source vertex should fail");

        assert_eq!(
            err,
            SurfaceDiscretizationError::CadProjectionOutsideFaceDomain {
                face_id: 7,
                node_id: 1,
            }
        );
    }

    #[test]
    fn centroid_subdivision_preserves_cad_face_ownership_and_boundary_edges() {
        let topology = single_triangle_topology();
        let cad_topology =
            build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

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
            build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = discretize_topology_curves(
            &topology,
            CurveDiscretizationOptions {
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

        assert_eq!(surface.elements.len(), 4);
        assert!(surface.nodes.len() > topology.vertices.len());
        assert!(surface
            .elements
            .iter()
            .all(|element| element.cad_face_id == Some("cad_face_7".to_string())));
        assert!(surface.elements.iter().any(|element| {
            element
                .source_edge_ids
                .iter()
                .any(|edge_id| *edge_id != INTERNAL_SOURCE_EDGE_ID)
        }));
        assert!(surface.elements.iter().any(|element| {
            element
                .source_edge_ids
                .iter()
                .any(|edge_id| *edge_id == INTERNAL_SOURCE_EDGE_ID)
        }));
    }

    #[test]
    fn curve_fan_fallback_orients_elements_to_cad_frame() {
        let face = single_triangle_topology().faces[0].clone();
        let frame = planar_test_frame(face.face_id);
        let mut nodes = vec![
            SurfaceNode {
                node_id: 0,
                source_vertex_id: 0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            SurfaceNode {
                node_id: 1,
                source_vertex_id: 1,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            SurfaceNode {
                node_id: 2,
                source_vertex_id: 2,
                coordinates_m: [0.0, 1.0, 0.0],
            },
        ];
        let segments = [
            FaceCurveSegment {
                node_ids: [1, 0],
                source_edge_id: 0,
            },
            FaceCurveSegment {
                node_ids: [2, 1],
                source_edge_id: 1,
            },
            FaceCurveSegment {
                node_ids: [0, 2],
                source_edge_id: 2,
            },
        ];
        let mut elements = Vec::<SurfaceElement>::new();

        append_curve_fan_face_elements(&face, &frame, &segments, &mut nodes, &mut elements);

        assert_eq!(elements.len(), 3);
        assert!(elements.iter().all(|element| {
            let points = element
                .node_ids
                .map(|node_id| nodes[node_id as usize].coordinates_m);
            dot(
                cross(sub(points[1], points[0]), sub(points[2], points[0])),
                frame.unit_normal,
            ) > 0.0
        }));
        for source_edge_id in [0, 1, 2] {
            assert!(elements
                .iter()
                .any(|element| element.source_edge_ids.contains(&source_edge_id)));
        }
    }

    #[test]
    fn curve_driven_cad_surface_uses_exact_face_domain_samples() {
        let topology = single_triangle_topology();
        let cad_topology = build_cad_topology(&geometry_with_face_domain_sample(), &topology)
            .expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        assert_eq!(cad_evaluation.report.evaluator_rejected_sample_count, 1);
        let curves = discretize_topology_curves(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 1.0,
                min_segments_per_edge: 1,
                max_segments_per_edge: 1,
            },
        )
        .expect("curves should discretize");

        let surface = discretize_cad_surfaces_with_curves(
            &topology,
            &cad_evaluation,
            &curves,
            SurfaceDiscretizationOptions {
                max_curve_segments_per_edge: 1,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("cad-owned curve surface should discretize");

        assert_eq!(surface.nodes.len(), topology.vertices.len() + 1);
        assert!(surface.elements.len() >= 2);
        assert_eq!(surface.exact_cad_sample_node_count, 1);
        assert_eq!(surface.rejected_exact_cad_sample_count, 0);
        assert!(surface
            .nodes
            .iter()
            .any(|node| node.coordinates_m == [0.25, 0.25, 0.0]));
        assert!(
            (surface
                .elements
                .iter()
                .map(|element| element.area_m2)
                .sum::<f64>()
                - topology.faces[0].area_m2)
                .abs()
                <= 1.0e-12
        );
        assert!(surface
            .elements
            .iter()
            .all(|element| element.cad_face_id == Some("face_a".to_string())));
    }

    #[test]
    fn curve_driven_cad_surface_preserves_area_with_multiple_exact_samples() {
        let topology = single_triangle_topology();
        let cad_topology =
            build_cad_topology(&geometry_with_area_regressing_face_samples(), &topology)
                .expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = discretize_topology_curves(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 1.0,
                min_segments_per_edge: 1,
                max_segments_per_edge: 1,
            },
        )
        .expect("curves should discretize");

        let surface = discretize_cad_surfaces_with_curves(
            &topology,
            &cad_evaluation,
            &curves,
            SurfaceDiscretizationOptions {
                max_curve_segments_per_edge: 1,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("cad-owned curve surface should discretize");
        let recovered_area = surface
            .elements
            .iter()
            .filter(|element| element.source_face_id == 7)
            .map(|element| element.area_m2)
            .sum::<f64>();

        assert_eq!(surface.nodes.len(), topology.vertices.len() + 3);
        assert_eq!(surface.elements.len(), 7);
        assert_eq!(surface.exact_cad_sample_node_count, 3);
        assert_eq!(surface.rejected_exact_cad_sample_count, 0);
        assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
        assert!(surface
            .elements
            .iter()
            .all(|element| element.cad_face_id == Some("face_a".to_string())));
    }

    #[test]
    fn curve_driven_cad_surface_splits_edge_hit_exact_samples_without_cracks() {
        let topology = single_triangle_topology();
        let cad_topology = build_cad_topology(&geometry_with_edge_hit_face_samples(), &topology)
            .expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = discretize_topology_curves(
            &topology,
            CurveDiscretizationOptions {
                target_size_m: 1.0,
                min_segments_per_edge: 1,
                max_segments_per_edge: 1,
            },
        )
        .expect("curves should discretize");

        let surface = discretize_cad_surfaces_with_curves(
            &topology,
            &cad_evaluation,
            &curves,
            SurfaceDiscretizationOptions {
                max_curve_segments_per_edge: 1,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("cad-owned curve surface should discretize");
        let recovered_area = surface
            .elements
            .iter()
            .filter(|element| element.source_face_id == 7)
            .map(|element| element.area_m2)
            .sum::<f64>();

        assert_eq!(surface.exact_cad_sample_node_count, 2);
        assert_eq!(surface.rejected_exact_cad_sample_count, 0);
        assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
        assert_local_surface_edges_are_recovered(&surface.elements);
    }

    #[test]
    fn curve_driven_cad_surface_rejects_samples_outside_concave_trim_loop() {
        let mut topology = single_triangle_topology();
        topology.faces[0].area_m2 = 0.275;
        let cad_topology =
            build_cad_topology(&geometry_with_concave_trim_rejected_sample(), &topology)
                .expect("cad topology");
        let cad_evaluation =
            build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = concave_trim_curve_discretization();

        let surface = discretize_cad_surfaces_with_curves(
            &topology,
            &cad_evaluation,
            &curves,
            SurfaceDiscretizationOptions {
                max_curve_segments_per_edge: 2,
                ..SurfaceDiscretizationOptions::default()
            },
        )
        .expect("concave trimmed surface should discretize");
        let recovered_area = surface
            .elements
            .iter()
            .filter(|element| element.source_face_id == 7)
            .map(|element| element.area_m2)
            .sum::<f64>();
        let trim_loop = [[0.0, 0.0], [0.5, 0.45], [1.0, 0.0], [0.0, 1.0]];

        assert_eq!(surface.exact_cad_sample_node_count, 0);
        assert_eq!(surface.rejected_exact_cad_sample_count, 1);
        assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
        assert!(!surface
            .nodes
            .iter()
            .any(|node| node.coordinates_m == [0.5, 0.2, 0.0]));
        assert!(surface.elements.iter().all(|element| {
            let centroid = triangle_centroid_2d(element.node_ids.map(|node_id| {
                let point = surface.nodes[node_id as usize].coordinates_m;
                [point[0], point[1]]
            }));
            point_in_polygon_2d(centroid, &trim_loop)
        }));
        assert_surface_edges_are_recovered(&surface.elements, &[[0, 3], [1, 3], [1, 2], [0, 2]]);
    }

    #[test]
    fn curve_driven_face_elements_triangulate_holed_loop_domain() {
        let face = SourceTopologyFace {
            face_id: 7,
            source_triangle_id: 11,
            node_ids: [0, 1, 2],
            edge_ids: [0, 1, 2],
            region_ids: vec!["face_a".to_string()],
            area_m2: 0.96,
            unit_normal: [0.0, 0.0, 1.0],
        };
        let frame = planar_test_frame(7);
        let mut nodes = square_with_square_hole_surface_nodes();
        let segment_loops = vec![
            vec![
                FaceCurveSegment {
                    node_ids: [0, 1],
                    source_edge_id: 0,
                },
                FaceCurveSegment {
                    node_ids: [1, 2],
                    source_edge_id: 1,
                },
                FaceCurveSegment {
                    node_ids: [2, 3],
                    source_edge_id: 2,
                },
                FaceCurveSegment {
                    node_ids: [3, 0],
                    source_edge_id: 3,
                },
            ],
            vec![
                FaceCurveSegment {
                    node_ids: [4, 5],
                    source_edge_id: 4,
                },
                FaceCurveSegment {
                    node_ids: [5, 6],
                    source_edge_id: 5,
                },
                FaceCurveSegment {
                    node_ids: [6, 7],
                    source_edge_id: 6,
                },
                FaceCurveSegment {
                    node_ids: [7, 4],
                    source_edge_id: 7,
                },
            ],
        ];
        let mut elements = Vec::<SurfaceElement>::new();

        let report = append_curve_driven_face_elements(
            &face,
            &frame,
            &segment_loops,
            &mut nodes,
            &mut elements,
        );
        let recovered_area = elements.iter().map(|element| element.area_m2).sum::<f64>();
        let hole = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]];

        assert_eq!(report, ExactCadSampleSurfaceReport::default());
        assert!(!elements.is_empty());
        assert!(
            (recovered_area - face.area_m2).abs() <= 1.0e-12,
            "recovered_area={recovered_area} expected_area={} element_count={}",
            face.area_m2,
            elements.len()
        );
        assert!(elements.iter().all(|element| {
            let centroid = triangle_centroid_2d(element.node_ids.map(|node_id| {
                let point = nodes[node_id as usize].coordinates_m;
                [point[0], point[1]]
            }));
            !point_in_polygon_2d(centroid, &hole)
        }));
        assert_surface_edges_are_recovered(
            &elements,
            &[
                [0, 1],
                [1, 2],
                [2, 3],
                [0, 3],
                [4, 5],
                [5, 6],
                [6, 7],
                [4, 7],
            ],
        );
    }

    #[test]
    fn single_loop_extractor_reports_multiple_face_curve_loops() {
        let segments = vec![
            FaceCurveSegment {
                node_ids: [0, 1],
                source_edge_id: 0,
            },
            FaceCurveSegment {
                node_ids: [1, 2],
                source_edge_id: 1,
            },
            FaceCurveSegment {
                node_ids: [2, 0],
                source_edge_id: 2,
            },
            FaceCurveSegment {
                node_ids: [3, 4],
                source_edge_id: 3,
            },
            FaceCurveSegment {
                node_ids: [4, 5],
                source_edge_id: 4,
            },
            FaceCurveSegment {
                node_ids: [5, 3],
                source_edge_id: 5,
            },
        ];

        let err = single_face_curve_segment_loop(7, &segments)
            .expect_err("multi-loop face topology should fail closed");

        assert_eq!(
            err,
            SurfaceDiscretizationError::MultipleFaceLoopsUnsupported {
                face_id: 7,
                loop_count: 2,
                loop_node_counts: vec![3, 3],
                loop_source_edge_ids: vec![vec![0, 1, 2], vec![3, 4, 5]],
            }
        );
        assert!(err
            .to_string()
            .contains("boundary loops with node counts [3, 3]"));
        assert!(err
            .to_string()
            .contains("source edge loops [[0, 1, 2], [3, 4, 5]]"));
    }

    #[test]
    fn face_curve_segment_loops_order_shuffled_single_loop_deterministically() {
        let segments = vec![
            FaceCurveSegment {
                node_ids: [3, 0],
                source_edge_id: 13,
            },
            FaceCurveSegment {
                node_ids: [1, 2],
                source_edge_id: 11,
            },
            FaceCurveSegment {
                node_ids: [2, 3],
                source_edge_id: 12,
            },
            FaceCurveSegment {
                node_ids: [0, 1],
                source_edge_id: 10,
            },
        ];

        let loops = face_curve_segment_loops(7, &segments).expect("loop should be valid");

        assert_eq!(loops.len(), 1);
        assert_eq!(
            loops[0],
            vec![
                FaceCurveSegment {
                    node_ids: [0, 1],
                    source_edge_id: 10,
                },
                FaceCurveSegment {
                    node_ids: [1, 2],
                    source_edge_id: 11,
                },
                FaceCurveSegment {
                    node_ids: [2, 3],
                    source_edge_id: 12,
                },
                FaceCurveSegment {
                    node_ids: [3, 0],
                    source_edge_id: 13,
                },
            ]
        );
    }

    #[test]
    fn rejects_open_face_curve_loop_before_triangulation() {
        let segments = vec![
            FaceCurveSegment {
                node_ids: [0, 1],
                source_edge_id: 0,
            },
            FaceCurveSegment {
                node_ids: [1, 2],
                source_edge_id: 1,
            },
        ];

        let err = single_face_curve_segment_loop(7, &segments)
            .expect_err("open face loop should fail closed");

        assert_eq!(
            err,
            SurfaceDiscretizationError::InvalidFaceLoopTopology {
                face_id: 7,
                node_id: 0,
                incident_segment_count: 1,
            }
        );
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

    fn planar_test_frame(source_face_id: u32) -> CadFaceEvaluationFrame {
        CadFaceEvaluationFrame {
            face_id: "face_a".to_string(),
            source_face_id,
            origin_m: [0.0, 0.0, 0.0],
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, 1.0, 0.0],
            unit_normal: [0.0, 0.0, 1.0],
            area_m2: 1.0,
            evaluator_backed: false,
            exact_query_backed: false,
            live_query_backed: false,
            evaluator_sample_count: 0,
            evaluator_rejected_sample_count: 0,
            evaluator_max_projection_error_m: 0.0,
            evaluator_samples: Vec::new(),
            u_derivative_m_per_uv: None,
            v_derivative_m_per_uv: None,
            max_curvature_estimate_1_per_m: None,
            uv_bounds: Some([[0.0, 0.0], [1.0, 1.0]]),
            uv_bounds_sample_count: 4,
            uv_domain_source: Some("test_domain".to_string()),
        }
    }

    fn square_with_square_hole_surface_nodes() -> Vec<SurfaceNode> {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.4, 0.4, 0.0],
            [0.6, 0.4, 0.0],
            [0.6, 0.6, 0.0],
            [0.4, 0.6, 0.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(node_id, coordinates_m)| SurfaceNode {
            node_id: node_id as u32,
            source_vertex_id: node_id as u32,
            coordinates_m,
        })
        .collect()
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

    fn geometry_with_face_domain_sample() -> runmat_geometry_core::GeometryAsset {
        let mut geometry = geometry_for_topology();
        geometry.regions = vec![Region {
            region_id: "face_a".to_string(),
            name: "face".to_string(),
            tag: Some("cad_face".to_string()),
            cad_ownership: Some(CadRegionOwnership {
                face_id: Some(7),
                label: Some(CadLabelRef {
                    label_entry: "0:1:7".to_string(),
                    name: "face".to_string(),
                    kind: CadSemanticKind::Face,
                }),
                owner_path: Vec::new(),
                layers: Vec::new(),
                color: None,
                material: None,
            }),
        }];
        geometry.region_entity_mappings = vec![RegionEntityMapping {
            region_id: "face_a".to_string(),
            mesh_id: "surface".to_string(),
            entity_kind: EntityKind::Face,
            ranges: vec![EntityIdRange {
                start: 11,
                count: 1,
            }],
        }];
        geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
            evaluator_id: "cad_evaluator_test".to_string(),
            backend: "test".to_string(),
            format_name: "step".to_string(),
            requires_source_geometry: true,
            faces: vec![CadFaceEvaluator {
                evaluator_id: "cad_face_7".to_string(),
                imported_face_id: 7,
                name: "face".to_string(),
                supports_point_evaluation: true,
                supports_projection: true,
                supports_normal: true,
                supports_derivatives: true,
                supports_curvature: true,
                reference_point_m: Some([0.25, 0.25, 0.0]),
                reference_unit_normal: Some([0.0, 0.0, 1.0]),
                evaluation_samples: vec![
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::BackendQuery,
                        point_m: [0.25, 0.25, 0.03],
                        uv: Some([0.25, 0.25]),
                        projected_point_m: Some([0.25, 0.25, 0.0]),
                        unit_normal: Some([0.0, 0.0, 1.0]),
                        projection_error_m: Some(0.03),
                    },
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::BackendQuery,
                        point_m: [1.25, 0.25, 0.0],
                        uv: Some([1.25, 0.25]),
                        projected_point_m: Some([1.25, 0.25, 0.0]),
                        unit_normal: Some([0.0, 0.0, 1.0]),
                        projection_error_m: Some(0.0),
                    },
                ],
            }],
            curves: Vec::new(),
        }];
        geometry
    }

    fn geometry_with_area_regressing_face_samples() -> runmat_geometry_core::GeometryAsset {
        let mut geometry = geometry_with_face_domain_sample();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.30, 0.10, 0.0],
                uv: Some([0.30, 0.10]),
                projected_point_m: Some([0.30, 0.10, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.75, 0.10, 0.0],
                uv: Some([0.75, 0.10]),
                projected_point_m: Some([0.75, 0.10, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.70, 0.25, 0.0],
                uv: Some([0.70, 0.25]),
                projected_point_m: Some([0.70, 0.25, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
        ];
        geometry
    }

    fn geometry_with_edge_hit_face_samples() -> runmat_geometry_core::GeometryAsset {
        let mut geometry = geometry_with_face_domain_sample();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.50, 0.25, 0.0],
                uv: Some([0.50, 0.25]),
                projected_point_m: Some([0.50, 0.25, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
            CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.25, 0.125, 0.0],
                uv: Some([0.25, 0.125]),
                projected_point_m: Some([0.25, 0.125, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            },
        ];
        geometry
    }

    fn geometry_with_concave_trim_rejected_sample() -> runmat_geometry_core::GeometryAsset {
        let mut geometry = geometry_with_face_domain_sample();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.2, 0.0],
                uv: Some([0.5, 0.2]),
                projected_point_m: Some([0.5, 0.2, 0.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.0),
            }];
        geometry
    }

    fn concave_trim_curve_discretization() -> CurveDiscretization {
        CurveDiscretization {
            nodes: vec![
                CurveNode {
                    node_id: 0,
                    source_edge_id: 0,
                    parameter: 0.0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                CurveNode {
                    node_id: 1,
                    source_edge_id: 0,
                    parameter: 0.5,
                    coordinates_m: [0.5, 0.45, 0.0],
                },
                CurveNode {
                    node_id: 2,
                    source_edge_id: 0,
                    parameter: 1.0,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
                CurveNode {
                    node_id: 3,
                    source_edge_id: 1,
                    parameter: 0.0,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
                CurveNode {
                    node_id: 4,
                    source_edge_id: 1,
                    parameter: 1.0,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
                CurveNode {
                    node_id: 5,
                    source_edge_id: 2,
                    parameter: 0.0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                CurveNode {
                    node_id: 6,
                    source_edge_id: 2,
                    parameter: 1.0,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
            ],
            elements: vec![
                CurveElement {
                    element_id: 0,
                    source_edge_id: 0,
                    node_ids: [0, 1],
                    length_m: 0.6726812023536856,
                },
                CurveElement {
                    element_id: 1,
                    source_edge_id: 0,
                    node_ids: [1, 2],
                    length_m: 0.6726812023536856,
                },
                CurveElement {
                    element_id: 2,
                    source_edge_id: 1,
                    node_ids: [3, 4],
                    length_m: 2.0_f64.sqrt(),
                },
                CurveElement {
                    element_id: 3,
                    source_edge_id: 2,
                    node_ids: [5, 6],
                    length_m: 1.0,
                },
            ],
        }
    }

    fn assert_local_surface_edges_are_recovered(elements: &[SurfaceElement]) {
        assert_surface_edges_are_recovered(elements, &[[0, 1], [0, 2], [1, 2]]);
    }

    fn assert_surface_edges_are_recovered(
        elements: &[SurfaceElement],
        boundary_edges: &[[u32; 2]],
    ) {
        let mut counts = BTreeMap::<[u32; 2], usize>::new();
        for element in elements {
            for edge in [
                sorted_node_pair(element.node_ids[0], element.node_ids[1]),
                sorted_node_pair(element.node_ids[1], element.node_ids[2]),
                sorted_node_pair(element.node_ids[2], element.node_ids[0]),
            ] {
                *counts.entry(edge).or_default() += 1;
            }
        }
        for (edge, count) in counts {
            let is_boundary = boundary_edges.contains(&edge);
            assert_eq!(
                count,
                if is_boundary { 1 } else { 2 },
                "unexpected local surface edge count for {edge:?}"
            );
        }
    }
}
