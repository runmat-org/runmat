use std::collections::{BTreeMap, BTreeSet};

use crate::{ExactFaceDelaunayError, ExactFaceDelaunayTriangle, ExactFacePslg};

use super::{ear::triangulate_polygon, recover::RecoveryControl};
use crate::exact_cdt::topology::{edge_uses, properly_crosses, sorted_edge};

/// Replaces the triangle strip intersected by `target` with two deterministic
/// polygon triangulations sharing `target`. Returning `false` means that the
/// intersected strip is not a single vertex-preserving disk cavity.
pub(super) fn recover_segment_cavity(
    triangles: &mut Vec<ExactFaceDelaunayTriangle>,
    pslg: &ExactFacePslg,
    target: [u32; 2],
    protected: &BTreeSet<[u32; 2]>,
    control: &mut RecoveryControl<'_>,
) -> Result<bool, ExactFaceDelaunayError> {
    control.checkpoint()?;
    let uses = edge_uses(triangles);
    let mut selected = BTreeSet::new();
    for (edge, edge_uses) in &uses {
        control.consume_predicates(4)?;
        if properly_crosses(*edge, target, pslg).map_err(|error| control.predicate_error(error))? {
            if edge_uses.len() != 2 || protected.contains(edge) {
                return Ok(false);
            }
            selected.extend(edge_uses.iter().map(|edge_use| edge_use.triangle_index));
        }
    }
    if selected.is_empty() {
        return Ok(false);
    }

    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    let mut selected_vertices = BTreeSet::new();
    for triangle_index in &selected {
        let vertices = triangles[*triangle_index].vertex_indices;
        selected_vertices.extend(vertices);
        for edge in [
            sorted_edge([vertices[0], vertices[1]]),
            sorted_edge([vertices[1], vertices[2]]),
            sorted_edge([vertices[2], vertices[0]]),
        ] {
            *edge_counts.entry(edge).or_default() += 1;
        }
    }
    if edge_counts.values().any(|count| *count > 2)
        || protected
            .iter()
            .any(|edge| edge_counts.get(edge).copied() == Some(2))
    {
        return Ok(false);
    }

    let boundary_edges = edge_counts
        .iter()
        .filter_map(|(edge, count)| (*count == 1).then_some(*edge))
        .collect::<Vec<_>>();
    let boundary_vertices = boundary_edges
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    if selected_vertices != boundary_vertices
        || !boundary_vertices.contains(&target[0])
        || !boundary_vertices.contains(&target[1])
    {
        return Ok(false);
    }

    let Some(cycle) = boundary_cycle(&boundary_edges, target[0]) else {
        return Ok(false);
    };
    let Some(target_position) = cycle.iter().position(|vertex| *vertex == target[1]) else {
        return Ok(false);
    };
    if target_position == 0 {
        return Ok(false);
    }

    let first_chain = cycle[..=target_position].to_vec();
    let mut second_chain = vec![target[0]];
    second_chain.extend(cycle[target_position..].iter().rev().copied());
    let mut replacement = triangulate_polygon(&first_chain, pslg, control)?;
    replacement.extend(triangulate_polygon(&second_chain, pslg, control)?);
    if replacement.len() != selected.len() {
        return Ok(false);
    }

    let replacement_edges = edge_uses(&replacement);
    if !replacement_edges.contains_key(&target)
        || protected
            .iter()
            .filter(|edge| edge_counts.contains_key(*edge))
            .any(|edge| !replacement_edges.contains_key(edge))
    {
        return Ok(false);
    }

    let retained_count = triangles.len().saturating_sub(selected.len());
    control.ensure_triangle_limit(retained_count.saturating_add(replacement.len()))?;
    control.consume_cavity()?;
    let mut triangle_index = 0usize;
    triangles.retain(|_| {
        let retain = !selected.contains(&triangle_index);
        triangle_index += 1;
        retain
    });
    triangles.extend(replacement);
    triangles.sort();
    if triangles.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(control.unsatisfied("recovery cavity produced a duplicate triangle"));
    }
    Ok(true)
}

fn boundary_cycle(boundary_edges: &[[u32; 2]], start: u32) -> Option<Vec<u32>> {
    let mut adjacency = BTreeMap::<u32, Vec<u32>>::new();
    for edge in boundary_edges {
        adjacency.entry(edge[0]).or_default().push(edge[1]);
        adjacency.entry(edge[1]).or_default().push(edge[0]);
    }
    if adjacency.is_empty() || adjacency.values().any(|neighbors| neighbors.len() != 2) {
        return None;
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_unstable();
    }

    let mut cycle = vec![start];
    let mut previous = None;
    let mut current = start;
    loop {
        let neighbors = adjacency.get(&current)?;
        let next = neighbors
            .iter()
            .copied()
            .find(|neighbor| Some(*neighbor) != previous)?;
        if next == start {
            break;
        }
        if cycle.contains(&next) {
            return None;
        }
        cycle.push(next);
        previous = Some(current);
        current = next;
    }
    (cycle.len() == adjacency.len()).then_some(cycle)
}
