use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_cad::{SourceTopologyEdge, SourceTopologyFace};
use runmat_meshing_curve::{CurveDiscretization, CurveNode};

use super::{SurfaceDiscretizationError, SurfaceNode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FaceCurveSegment {
    pub(super) node_ids: [u32; 2],
    pub(super) source_edge_id: u32,
}

pub(super) fn curve_nodes_by_source_edge(
    curves: &CurveDiscretization,
) -> BTreeMap<u32, Vec<&CurveNode>> {
    let mut by_edge = BTreeMap::<u32, Vec<&CurveNode>>::new();
    for node in &curves.nodes {
        by_edge.entry(node.source_edge_id).or_default().push(node);
    }
    for nodes in by_edge.values_mut() {
        nodes.sort_by(|left, right| left.parameter.total_cmp(&right.parameter));
    }
    by_edge
}

pub(super) fn oriented_face_curve_segments(
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

pub(super) fn curve_segments_for_source_edges(
    topology_edges: &BTreeMap<u32, &SourceTopologyEdge>,
    curve_nodes_by_edge: &BTreeMap<u32, Vec<&CurveNode>>,
    face_id: u32,
    source_edge_ids: &[u32],
    max_curve_segments_per_edge: usize,
    nodes: &mut Vec<SurfaceNode>,
    curve_node_to_surface_node: &mut BTreeMap<u32, u32>,
) -> Result<Vec<FaceCurveSegment>, SurfaceDiscretizationError> {
    let mut segments = Vec::<FaceCurveSegment>::new();
    for source_edge_id in source_edge_ids {
        let edge = topology_edges.get(source_edge_id).ok_or(
            SurfaceDiscretizationError::MissingFaceEdge {
                face_id,
                edge_id: *source_edge_id,
            },
        )?;
        let curve_nodes = curve_nodes_by_edge.get(source_edge_id).ok_or(
            SurfaceDiscretizationError::MissingCurveEdge {
                source_edge_id: *source_edge_id,
            },
        )?;
        let capped_nodes =
            capped_curve_nodes(curve_nodes.clone(), max_curve_segments_per_edge.max(1));
        for pair in capped_nodes.windows(2) {
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
                    source_edge_id: *source_edge_id,
                });
            }
        }
    }
    Ok(segments)
}

pub(super) fn face_curve_segment_loops(
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
