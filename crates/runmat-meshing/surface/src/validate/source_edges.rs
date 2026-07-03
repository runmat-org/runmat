use std::collections::{BTreeMap, BTreeSet};

use crate::{SurfaceDiscretization, INTERNAL_SOURCE_EDGE_ID};
use runmat_meshing_cad::SourceTopologyEdge;

use super::{geometry::sorted_edge, types::SurfaceValidationError};

pub(super) fn count_closed_source_edge_loops(
    edges: &[SourceTopologyEdge],
) -> Result<(usize, usize), SurfaceValidationError> {
    let mut endpoint_incidence = BTreeMap::<u32, usize>::new();
    let mut endpoint_edges = BTreeMap::<u32, Vec<u32>>::new();
    let mut edges_by_id = BTreeMap::<u32, &SourceTopologyEdge>::new();
    for edge in edges {
        edges_by_id.insert(edge.edge_id, edge);
        *endpoint_incidence.entry(edge.node_ids[0]).or_default() += 1;
        *endpoint_incidence.entry(edge.node_ids[1]).or_default() += 1;
        endpoint_edges
            .entry(edge.node_ids[0])
            .or_default()
            .push(edge.edge_id);
        endpoint_edges
            .entry(edge.node_ids[1])
            .or_default()
            .push(edge.edge_id);
    }

    let mut visited_edges = BTreeSet::<u32>::new();
    let mut component_count = 0_usize;
    let mut closed_count = 0_usize;
    for edge in edges {
        if !visited_edges.insert(edge.edge_id) {
            continue;
        }

        component_count += 1;
        let mut closed = true;
        let mut component_edge_ids = Vec::<u32>::new();
        let mut stack = vec![edge.edge_id];
        while let Some(edge_id) = stack.pop() {
            component_edge_ids.push(edge_id);
            let component_edge =
                edges_by_id
                    .get(&edge_id)
                    .ok_or(SurfaceValidationError::MissingSourceEdge {
                        source_edge_id: edge_id,
                    })?;
            for endpoint_id in component_edge.node_ids {
                let incidence_count = endpoint_incidence.get(&endpoint_id).copied().unwrap_or(0);
                if incidence_count < 2 {
                    closed = false;
                }
                if let Some(adjacent_edges) = endpoint_edges.get(&endpoint_id) {
                    for adjacent_edge_id in adjacent_edges {
                        if visited_edges.insert(*adjacent_edge_id) {
                            stack.push(*adjacent_edge_id);
                        }
                    }
                }
            }
        }

        if closed {
            closed_count += 1;
        } else {
            for endpoint_id in edge.node_ids {
                let incidence_count = endpoint_incidence.get(&endpoint_id).copied().unwrap_or(0);
                if incidence_count < 2 {
                    return Err(SurfaceValidationError::OpenSourceLoop {
                        source_edge_id: edge.edge_id,
                        endpoint_id,
                        incidence_count,
                    });
                }
            }
            for edge_id in component_edge_ids {
                let component_edge =
                    edges_by_id
                        .get(&edge_id)
                        .ok_or(SurfaceValidationError::MissingSourceEdge {
                            source_edge_id: edge_id,
                        })?;
                for endpoint_id in component_edge.node_ids {
                    let incidence_count =
                        endpoint_incidence.get(&endpoint_id).copied().unwrap_or(0);
                    if incidence_count < 2 {
                        return Err(SurfaceValidationError::OpenSourceLoop {
                            source_edge_id: edge_id,
                            endpoint_id,
                            incidence_count,
                        });
                    }
                }
            }
        }
    }

    Ok((component_count, closed_count))
}

pub(super) fn surface_edge_source_ids(
    surface: &SurfaceDiscretization,
) -> BTreeMap<u32, Vec<[u32; 2]>> {
    let mut edges = BTreeMap::<u32, Vec<[u32; 2]>>::new();
    for element in &surface.elements {
        for (source_edge_id, node_ids) in element.source_edge_ids.into_iter().zip([
            sorted_edge(element.node_ids[0], element.node_ids[1]),
            sorted_edge(element.node_ids[1], element.node_ids[2]),
            sorted_edge(element.node_ids[2], element.node_ids[0]),
        ]) {
            if source_edge_id != INTERNAL_SOURCE_EDGE_ID {
                edges.entry(source_edge_id).or_default().push(node_ids);
            }
        }
    }
    edges
}

pub(super) fn source_edge_is_recovered_by_chain(
    segments: &[[u32; 2]],
    source_edge: &SourceTopologyEdge,
) -> bool {
    let source_edge_nodes = sorted_edge(source_edge.node_ids[0], source_edge.node_ids[1]);
    if segments.contains(&source_edge_nodes) {
        return true;
    }
    let mut adjacency = BTreeMap::<u32, Vec<u32>>::new();
    for segment in segments {
        adjacency.entry(segment[0]).or_default().push(segment[1]);
        adjacency.entry(segment[1]).or_default().push(segment[0]);
    }
    let mut stack = vec![source_edge.node_ids[0]];
    let mut visited = BTreeSet::<u32>::new();
    while let Some(node_id) = stack.pop() {
        if !visited.insert(node_id) {
            continue;
        }
        if node_id == source_edge.node_ids[1] {
            return true;
        }
        if let Some(next) = adjacency.get(&node_id) {
            stack.extend(next.iter().copied());
        }
    }
    false
}
