use std::collections::BTreeMap;

use runmat_geometry_core::CadFaceEvaluationSampleSource;
use serde::{Deserialize, Serialize};

use crate::{
    cad_eval::{project_to_face, CadEvaluationModel},
    curve::{CurveDiscretization, CurveNode},
    predicate::{cross, dot, sub, triangle_area},
    source_topology::{SourceTopologyFace, SourceTopologyModel},
};

pub const INTERNAL_SOURCE_EDGE_ID: u32 = u32::MAX;
const FACE_AREA_RECOVERY_TOLERANCE: f64 = 1.0e-8;

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
        let sample_report =
            append_curve_driven_face_elements(face, frame, &segments, &mut nodes, &mut elements);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaceTriangle {
    point_indices: [usize; 3],
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

fn append_curve_driven_face_elements(
    face: &SourceTopologyFace,
    frame: &crate::CadFaceEvaluationFrame,
    segments: &[FaceCurveSegment],
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) -> ExactCadSampleSurfaceReport {
    if segments.len() <= 3 && !has_exact_face_domain_samples(frame) {
        append_curve_fan_face_elements(face, frame, segments, nodes, elements);
        return ExactCadSampleSurfaceReport::default();
    }

    let node_start = nodes.len();
    let element_start = elements.len();
    let mut boundary_edge_ids = BTreeMap::<[u32; 2], u32>::new();
    for segment in segments {
        boundary_edge_ids.insert(
            sorted_node_pair(segment.node_ids[0], segment.node_ids[1]),
            segment.source_edge_id,
        );
    }

    let mut points = boundary_triangulation_points(frame, segments, nodes);
    let boundary_point_count = points.len();
    let boundary_polygon = boundary_loop_polygon(&points[..boundary_point_count]);
    let sample_report =
        append_exact_face_domain_sample_points(face, frame, &boundary_polygon, nodes, &mut points);
    append_face_lattice_points(face, frame, &boundary_polygon, segments, nodes, &mut points);
    let triangles = if boundary_point_count == 3 {
        triangulate_triangle_points_by_insertion(&points, boundary_point_count)
    } else {
        triangulate_face_points(&points, &boundary_polygon)
    };
    if triangles.is_empty() {
        append_curve_fan_face_elements(face, frame, segments, nodes, elements);
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
        append_curve_fan_face_elements(face, frame, segments, nodes, elements);
        return sample_report.rejected_after_area_guard();
    }
    sample_report
}

fn append_curve_fan_face_elements(
    face: &SourceTopologyFace,
    frame: &crate::CadFaceEvaluationFrame,
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

fn boundary_triangulation_points(
    frame: &crate::CadFaceEvaluationFrame,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ExactCadSampleSurfaceReport {
    accepted_count: usize,
    rejected_count: usize,
}

impl ExactCadSampleSurfaceReport {
    fn rejected_after_area_guard(self) -> Self {
        Self {
            accepted_count: 0,
            rejected_count: self.rejected_count + self.accepted_count,
        }
    }
}

fn face_area_is_recovered(face: &SourceTopologyFace, elements: &[SurfaceElement]) -> bool {
    if elements.is_empty() {
        return false;
    }
    let recovered_area_m2 = elements
        .iter()
        .filter(|element| element.source_face_id == face.face_id)
        .map(|element| element.area_m2)
        .sum::<f64>();
    if !recovered_area_m2.is_finite() || !face.area_m2.is_finite() || face.area_m2 <= 0.0 {
        return false;
    }
    ((recovered_area_m2 - face.area_m2).abs() / face.area_m2) <= FACE_AREA_RECOVERY_TOLERANCE
}

fn face_edges_are_recovered(
    elements: &[SurfaceElement],
    boundary_edge_ids: &BTreeMap<[u32; 2], u32>,
) -> bool {
    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    for element in elements {
        for edge in [
            sorted_node_pair(element.node_ids[0], element.node_ids[1]),
            sorted_node_pair(element.node_ids[1], element.node_ids[2]),
            sorted_node_pair(element.node_ids[2], element.node_ids[0]),
        ] {
            *edge_counts.entry(edge).or_default() += 1;
        }
    }
    edge_counts.into_iter().all(|(edge, count)| {
        if boundary_edge_ids.contains_key(&edge) {
            count == 1
        } else {
            count == 2
        }
    })
}

fn append_exact_face_domain_sample_points(
    face: &SourceTopologyFace,
    frame: &crate::CadFaceEvaluationFrame,
    boundary_polygon: &[[f64; 2]],
    nodes: &mut Vec<SurfaceNode>,
    points: &mut Vec<FaceTriangulationPoint>,
) -> ExactCadSampleSurfaceReport {
    let mut report = ExactCadSampleSurfaceReport::default();
    let face_points = face
        .node_ids
        .map(|node_id| nodes[node_id as usize].coordinates_m);
    for sample in &frame.evaluator_samples {
        if !is_usable_exact_face_domain_sample(sample) {
            report.rejected_count += 1;
            continue;
        }
        let coordinates = sample
            .projected_point_m
            .filter(|point| finite_point3(*point))
            .unwrap_or(sample.point_m);
        let projection = project_to_face(frame, coordinates);
        if !point_in_triangle_3d(projection.point_m, face_points) {
            report.rejected_count += 1;
            continue;
        }
        let local_uv = frame_local_uv(frame, projection.point_m);
        if !point_in_polygon_2d(local_uv, boundary_polygon) {
            report.rejected_count += 1;
            continue;
        }
        if points
            .iter()
            .any(|point| distance2_2d(point.uv, local_uv) <= 1.0e-24)
        {
            report.rejected_count += 1;
            continue;
        }
        let node_id = nodes.len() as u32;
        nodes.push(SurfaceNode {
            node_id,
            source_vertex_id: u32::MAX,
            coordinates_m: projection.point_m,
        });
        points.push(FaceTriangulationPoint {
            node_id,
            uv: local_uv,
        });
        report.accepted_count += 1;
    }
    report
}

fn frame_local_uv(frame: &crate::CadFaceEvaluationFrame, point_m: [f64; 3]) -> [f64; 2] {
    let relative = sub(point_m, frame.origin_m);
    [dot(relative, frame.u_axis), dot(relative, frame.v_axis)]
}

fn has_exact_face_domain_samples(frame: &crate::CadFaceEvaluationFrame) -> bool {
    frame
        .evaluator_samples
        .iter()
        .any(is_usable_exact_face_domain_sample)
}

fn is_usable_exact_face_domain_sample(
    sample: &runmat_geometry_core::CadFaceEvaluationSample,
) -> bool {
    sample.source == CadFaceEvaluationSampleSource::BackendQuery
        && finite_point3(sample.point_m)
        && sample.uv.is_some_and(finite_point2)
        && sample.projected_point_m.is_none_or(finite_point3)
}

fn append_face_lattice_points(
    face: &SourceTopologyFace,
    frame: &crate::CadFaceEvaluationFrame,
    boundary_polygon: &[[f64; 2]],
    segments: &[FaceCurveSegment],
    nodes: &mut Vec<SurfaceNode>,
    points: &mut Vec<FaceTriangulationPoint>,
) {
    let segment_count = segments_per_source_edge(segments)
        .into_values()
        .max()
        .unwrap_or(1)
        .max(2);
    if segment_count <= 2 {
        return;
    }
    let corners = face
        .node_ids
        .map(|node_id| nodes[node_id as usize].coordinates_m);
    for i in 1..segment_count {
        for j in 1..(segment_count - i) {
            let u = i as f64 / segment_count as f64;
            let v = j as f64 / segment_count as f64;
            let w = 1.0 - u - v;
            if w <= f64::EPSILON {
                continue;
            }
            let coordinates = [
                corners[0][0] * w + corners[1][0] * u + corners[2][0] * v,
                corners[0][1] * w + corners[1][1] * u + corners[2][1] * v,
                corners[0][2] * w + corners[1][2] * u + corners[2][2] * v,
            ];
            let projection = project_to_face(frame, coordinates);
            if !point_in_polygon_2d(projection.uv, boundary_polygon) {
                continue;
            }
            if points
                .iter()
                .any(|point| distance2_2d(point.uv, projection.uv) <= 1.0e-24)
            {
                continue;
            }
            let node_id = nodes.len() as u32;
            nodes.push(SurfaceNode {
                node_id,
                source_vertex_id: u32::MAX,
                coordinates_m: projection.point_m,
            });
            points.push(FaceTriangulationPoint {
                node_id,
                uv: projection.uv,
            });
        }
    }
}

fn segments_per_source_edge(segments: &[FaceCurveSegment]) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::<u32, usize>::new();
    for segment in segments {
        *counts.entry(segment.source_edge_id).or_default() += 1;
    }
    counts
}

fn triangulate_face_points(
    points: &[FaceTriangulationPoint],
    boundary_polygon: &[[f64; 2]],
) -> Vec<FaceTriangle> {
    if points.len() < 3 {
        return Vec::new();
    }
    let mut work_points = points
        .iter()
        .map(|point| TriangulationPoint {
            uv: point.uv,
            original_index: Some(0),
            is_super: false,
        })
        .collect::<Vec<_>>();
    for (index, point) in work_points.iter_mut().enumerate() {
        point.original_index = Some(index);
    }
    let super_start = work_points.len();
    work_points.extend(super_triangle_points(points));
    let mut triangles = vec![TriangulationTriangle {
        point_indices: [super_start, super_start + 1, super_start + 2],
    }];

    for point_index in 0..points.len() {
        let point = work_points[point_index].uv;
        let mut bad_indices = Vec::<usize>::new();
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            if circumcircle_contains(
                triangle.point_indices.map(|index| work_points[index].uv),
                point,
            ) {
                bad_indices.push(triangle_index);
            }
        }
        if bad_indices.is_empty() {
            continue;
        }
        let bad_set = bad_indices
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let mut edge_counts = BTreeMap::<[usize; 2], usize>::new();
        for triangle_index in &bad_indices {
            for edge in triangle_edges_2d(triangles[*triangle_index].point_indices) {
                *edge_counts
                    .entry(sorted_index_pair(edge[0], edge[1]))
                    .or_default() += 1;
            }
        }
        let cavity_edges = edge_counts
            .into_iter()
            .filter_map(|(edge, count)| (count == 1).then_some(edge))
            .collect::<Vec<_>>();
        triangles = triangles
            .into_iter()
            .enumerate()
            .filter_map(|(index, triangle)| (!bad_set.contains(&index)).then_some(triangle))
            .collect();
        for edge in cavity_edges {
            let point_indices = [edge[0], edge[1], point_index];
            if triangle_area_2d(point_indices.map(|index| work_points[index].uv)).abs()
                > f64::EPSILON
            {
                triangles.push(TriangulationTriangle { point_indices });
            }
        }
    }

    triangles
        .into_iter()
        .filter(|triangle| {
            !triangle
                .point_indices
                .iter()
                .any(|index| work_points[*index].is_super)
        })
        .filter_map(|triangle| {
            let point_indices = triangle
                .point_indices
                .map(|index| work_points[index].original_index);
            Some(FaceTriangle {
                point_indices: [point_indices[0]?, point_indices[1]?, point_indices[2]?],
            })
        })
        .filter(|triangle| {
            let centroid =
                triangle_centroid_2d(triangle.point_indices.map(|index| points[index].uv));
            point_in_polygon_2d(centroid, boundary_polygon)
        })
        .collect()
}

fn triangulate_triangle_points_by_insertion(
    points: &[FaceTriangulationPoint],
    boundary_point_count: usize,
) -> Vec<FaceTriangle> {
    if boundary_point_count != 3 || points.len() < 3 {
        return Vec::new();
    }
    let mut triangles = vec![FaceTriangle {
        point_indices: [0, 1, 2],
    }];
    for point_index in boundary_point_count..points.len() {
        let edge_hits = triangles
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(triangle_index, triangle)| {
                triangle_edge_containing_point(point_index, triangle, points)
                    .map(|edge| (triangle_index, triangle, edge))
            })
            .collect::<Vec<_>>();
        if !edge_hits.is_empty() {
            for (triangle_index, triangle, edge) in edge_hits.into_iter().rev() {
                triangles.swap_remove(triangle_index);
                let opposite = triangle
                    .point_indices
                    .into_iter()
                    .find(|index| *index != edge[0] && *index != edge[1])
                    .expect("triangle edge should have an opposite point");
                push_non_degenerate_face_triangle(
                    &mut triangles,
                    [edge[0], point_index, opposite],
                    points,
                );
                push_non_degenerate_face_triangle(
                    &mut triangles,
                    [point_index, edge[1], opposite],
                    points,
                );
            }
            continue;
        }
        let Some((triangle_index, triangle)) =
            triangles.iter().copied().enumerate().find(|(_, triangle)| {
                point_in_triangle_2d(
                    points[point_index].uv,
                    triangle.point_indices.map(|index| points[index].uv),
                )
            })
        else {
            continue;
        };
        triangles.swap_remove(triangle_index);
        for point_indices in [
            [
                triangle.point_indices[0],
                triangle.point_indices[1],
                point_index,
            ],
            [
                triangle.point_indices[1],
                triangle.point_indices[2],
                point_index,
            ],
            [
                triangle.point_indices[2],
                triangle.point_indices[0],
                point_index,
            ],
        ] {
            push_non_degenerate_face_triangle(&mut triangles, point_indices, points);
        }
    }
    triangles
}

fn triangle_edge_containing_point(
    point_index: usize,
    triangle: FaceTriangle,
    points: &[FaceTriangulationPoint],
) -> Option<[usize; 2]> {
    let point = points[point_index].uv;
    for edge in triangle_edges_2d(triangle.point_indices) {
        if point_on_segment_2d(point, points[edge[0]].uv, points[edge[1]].uv) {
            return Some(edge);
        }
    }
    None
}

fn push_non_degenerate_face_triangle(
    triangles: &mut Vec<FaceTriangle>,
    point_indices: [usize; 3],
    points: &[FaceTriangulationPoint],
) {
    if triangle_area_2d(point_indices.map(|index| points[index].uv)).abs() > f64::EPSILON {
        triangles.push(FaceTriangle { point_indices });
    }
}

fn point_in_triangle_2d(point: [f64; 2], triangle: [[f64; 2]; 3]) -> bool {
    let area = triangle_area_2d(triangle);
    if area.abs() <= f64::EPSILON {
        return false;
    }
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let edge_areas = [
        triangle_area_2d([triangle[0], triangle[1], point]) * sign,
        triangle_area_2d([triangle[1], triangle[2], point]) * sign,
        triangle_area_2d([triangle[2], triangle[0], point]) * sign,
    ];
    edge_areas.iter().all(|value| *value >= -1.0e-12)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TriangulationPoint {
    uv: [f64; 2],
    original_index: Option<usize>,
    is_super: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TriangulationTriangle {
    point_indices: [usize; 3],
}

fn super_triangle_points(points: &[FaceTriangulationPoint]) -> [TriangulationPoint; 3] {
    let mut min = points[0].uv;
    let mut max = points[0].uv;
    for point in points {
        min[0] = min[0].min(point.uv[0]);
        min[1] = min[1].min(point.uv[1]);
        max[0] = max[0].max(point.uv[0]);
        max[1] = max[1].max(point.uv[1]);
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
    let span = (max[0] - min[0]).max(max[1] - min[1]).max(1.0);
    [
        TriangulationPoint {
            uv: [center[0] - 32.0 * span, center[1] - span],
            original_index: None,
            is_super: true,
        },
        TriangulationPoint {
            uv: [center[0], center[1] + 32.0 * span],
            original_index: None,
            is_super: true,
        },
        TriangulationPoint {
            uv: [center[0] + 32.0 * span, center[1] - span],
            original_index: None,
            is_super: true,
        },
    ]
}

fn circumcircle_contains(triangle: [[f64; 2]; 3], point: [f64; 2]) -> bool {
    let ax = triangle[0][0] - point[0];
    let ay = triangle[0][1] - point[1];
    let bx = triangle[1][0] - point[0];
    let by = triangle[1][1] - point[1];
    let cx = triangle[2][0] - point[0];
    let cy = triangle[2][1] - point[1];
    let determinant = (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx);
    let orientation = triangle_area_2d(triangle);
    if orientation > 0.0 {
        determinant > -1.0e-12
    } else {
        determinant < 1.0e-12
    }
}

fn triangle_edges_2d(point_indices: [usize; 3]) -> [[usize; 2]; 3] {
    [
        [point_indices[0], point_indices[1]],
        [point_indices[1], point_indices[2]],
        [point_indices[2], point_indices[0]],
    ]
}

fn triangle_area_2d(points: [[f64; 2]; 3]) -> f64 {
    0.5 * ((points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
        - (points[1][1] - points[0][1]) * (points[2][0] - points[0][0]))
}

fn triangle_centroid_2d(points: [[f64; 2]; 3]) -> [f64; 2] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
    ]
}

fn boundary_loop_polygon(points: &[FaceTriangulationPoint]) -> Vec<[f64; 2]> {
    let mut polygon = Vec::<[f64; 2]>::new();
    for point in points {
        if polygon
            .last()
            .is_some_and(|last| distance2_2d(*last, point.uv) <= 1.0e-24)
        {
            continue;
        }
        polygon.push(point.uv);
    }
    if polygon.len() > 1
        && distance2_2d(
            polygon[0],
            *polygon.last().expect("polygon should be non-empty"),
        ) <= 1.0e-24
    {
        polygon.pop();
    }
    polygon
}

fn point_in_polygon_2d(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for current in polygon {
        if point_on_segment_2d(point, previous, *current) {
            return true;
        }
        let denominator = previous[1] - current[1];
        let crosses = denominator.abs() > f64::EPSILON
            && ((current[1] > point[1]) != (previous[1] > point[1]))
            && point[0]
                < (previous[0] - current[0]) * (point[1] - current[1]) / denominator + current[0];
        if crosses {
            inside = !inside;
        }
        previous = *current;
    }
    inside
}

fn point_on_segment_2d(point: [f64; 2], start: [f64; 2], end: [f64; 2]) -> bool {
    cross_2d(start, end, point).abs() <= 1.0e-10
        && point[0] >= start[0].min(end[0]) - 1.0e-10
        && point[0] <= start[0].max(end[0]) + 1.0e-10
        && point[1] >= start[1].min(end[1]) - 1.0e-10
        && point[1] <= start[1].max(end[1]) + 1.0e-10
}

fn cross_2d(origin: [f64; 2], left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])
}

fn distance2_2d(left: [f64; 2], right: [f64; 2]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    dx * dx + dy * dy
}

fn finite_point2(point: [f64; 2]) -> bool {
    point.iter().all(|value| value.is_finite())
}

fn finite_point3(point: [f64; 3]) -> bool {
    point.iter().all(|value| value.is_finite())
}

fn point_in_triangle_3d(point: [f64; 3], triangle: [[f64; 3]; 3]) -> bool {
    let v0 = sub(triangle[2], triangle[0]);
    let v1 = sub(triangle[1], triangle[0]);
    let v2 = sub(point, triangle[0]);
    let dot00 = dot(v0, v0);
    let dot01 = dot(v0, v1);
    let dot02 = dot(v0, v2);
    let dot11 = dot(v1, v1);
    let dot12 = dot(v1, v2);
    let denominator = dot00 * dot11 - dot01 * dot01;
    if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return false;
    }
    let inv_denominator = 1.0 / denominator;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denominator;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denominator;
    let tolerance = 1.0e-10;
    u >= -tolerance && v >= -tolerance && u + v <= 1.0 + tolerance
}

fn sorted_node_pair(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn sorted_index_pair(left: usize, right: usize) -> [usize; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
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
    use runmat_geometry_core::{
        CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
        CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, Region,
        RegionEntityMapping,
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
    fn curve_driven_cad_surface_uses_exact_face_domain_samples() {
        let topology = single_triangle_topology();
        let cad_topology =
            crate::build_cad_topology(&geometry_with_face_domain_sample(), &topology)
                .expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = crate::discretize_topology_curves(
            &topology,
            crate::CurveDiscretizationOptions {
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
        assert_eq!(surface.rejected_exact_cad_sample_count, 1);
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
            crate::build_cad_topology(&geometry_with_area_regressing_face_samples(), &topology)
                .expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = crate::discretize_topology_curves(
            &topology,
            crate::CurveDiscretizationOptions {
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
        let cad_topology =
            crate::build_cad_topology(&geometry_with_edge_hit_face_samples(), &topology)
                .expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
        let curves = crate::discretize_topology_curves(
            &topology,
            crate::CurveDiscretizationOptions {
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
            crate::build_cad_topology(&geometry_with_concave_trim_rejected_sample(), &topology)
                .expect("cad topology");
        let cad_evaluation =
            crate::build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
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

    fn concave_trim_curve_discretization() -> crate::curve::CurveDiscretization {
        crate::curve::CurveDiscretization {
            nodes: vec![
                crate::curve::CurveNode {
                    node_id: 0,
                    source_edge_id: 0,
                    parameter: 0.0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                crate::curve::CurveNode {
                    node_id: 1,
                    source_edge_id: 0,
                    parameter: 0.5,
                    coordinates_m: [0.5, 0.45, 0.0],
                },
                crate::curve::CurveNode {
                    node_id: 2,
                    source_edge_id: 0,
                    parameter: 1.0,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
                crate::curve::CurveNode {
                    node_id: 3,
                    source_edge_id: 1,
                    parameter: 0.0,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
                crate::curve::CurveNode {
                    node_id: 4,
                    source_edge_id: 1,
                    parameter: 1.0,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
                crate::curve::CurveNode {
                    node_id: 5,
                    source_edge_id: 2,
                    parameter: 0.0,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                crate::curve::CurveNode {
                    node_id: 6,
                    source_edge_id: 2,
                    parameter: 1.0,
                    coordinates_m: [0.0, 1.0, 0.0],
                },
            ],
            elements: vec![
                crate::curve::CurveElement {
                    element_id: 0,
                    source_edge_id: 0,
                    node_ids: [0, 1],
                    length_m: 0.6726812023536856,
                },
                crate::curve::CurveElement {
                    element_id: 1,
                    source_edge_id: 0,
                    node_ids: [1, 2],
                    length_m: 0.6726812023536856,
                },
                crate::curve::CurveElement {
                    element_id: 2,
                    source_edge_id: 1,
                    node_ids: [3, 4],
                    length_m: 2.0_f64.sqrt(),
                },
                crate::curve::CurveElement {
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
