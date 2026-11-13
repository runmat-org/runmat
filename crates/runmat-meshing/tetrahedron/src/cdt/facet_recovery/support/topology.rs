//! Deterministic polygon support over the actual local topology graph.
//! An interval can be reachable only if its endpoint edge exists. Visit those
//! intervals and candidate common neighbors, rather than a dense cubic table.
use super::*;

pub(super) fn triangulate_with_existing_topology_edges(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    boundary: &[StableDigest],
    axes: [usize; 2],
    polygon_sign: PredicateSign,
    source_edge_membership: &BTreeMap<StableDigest, BTreeSet<usize>>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<DelaunayRecoveredFacetTriangle>>, DelaunayFacetRecoveryError> {
    let neighbors = boundary_neighbors(recovery, boundary, constraint_index, work)?;
    let mut intervals = Vec::new();
    let mut splits = BTreeMap::new();
    for (first, adjacent) in neighbors.iter().enumerate().take(boundary.len() - 1) {
        splits.insert((first, first + 1), usize::MAX);
        for &last in adjacent.range(first + 2..) {
            work.support_step(constraint_index)?;
            intervals.push((last - first, first, last));
        }
    }
    intervals.sort_unstable();
    for (_, first, last) in intervals {
        // Ascending middle indices preserve the dense algorithm's tie breaking.
        for &middle in neighbors[first].range(first + 1..last) {
            work.support_step(constraint_index)?;
            if !neighbors[last].contains(&middle)
                || !splits.contains_key(&(first, middle))
                || !splits.contains_key(&(middle, last))
            {
                continue;
            }
            let triangle = [boundary[first], boundary[middle], boundary[last]];
            if shares_source_edge(triangle, source_edge_membership)
                || orientation(recovery, triangle, axes, constraint_index, work)? != polygon_sign
            {
                continue;
            }
            splits.insert((first, last), middle);
            break;
        }
    }
    if !splits.contains_key(&(0, boundary.len() - 1)) {
        return Ok(None);
    }
    let mut triangles = Vec::with_capacity(boundary.len() - 2);
    let mut pending = vec![(0, boundary.len() - 1)];
    while let Some((first, last)) = pending.pop() {
        work.support_step(constraint_index)?;
        if last == first + 1 {
            continue;
        }
        let middle = *splits.get(&(first, last)).ok_or_else(|| {
            invalid(
                constraint_index,
                "facet support lost a reachable interval split",
            )
        })?;
        triangles.push(DelaunayRecoveredFacetTriangle {
            node_identities: [boundary[first], boundary[middle], boundary[last]],
        });
        pending.push((middle, last));
        pending.push((first, middle));
    }
    Ok(Some(triangles))
}

fn boundary_neighbors(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    boundary: &[StableDigest],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Vec<BTreeSet<usize>>, DelaunayFacetRecoveryError> {
    let mut indices = BTreeMap::new();
    for (position, identity) in boundary.iter().enumerate() {
        work.support_step(constraint_index)?;
        let index = node_index(recovery.topology, *identity).ok_or_else(|| {
            invalid(
                constraint_index,
                "facet support node is missing from topology",
            )
        })?;
        indices.insert(index as u32, position);
    }
    let mut neighbors = vec![BTreeSet::new(); boundary.len()];
    let mut visited = BTreeSet::new();
    for &vertex in indices.keys() {
        for tetrahedron in recovery.topology.vertex_star(vertex) {
            work.support_step(constraint_index)?;
            if !visited.insert(tetrahedron) {
                continue;
            }
            let positions = recovery
                .topology
                .tetrahedron(tetrahedron)
                .expect("live facet support star")
                .vertex_indices
                .map(|other| indices.get(&other).copied());
            // One charged incidence visit has a fixed four-corner inventory.
            for position in positions.into_iter().flatten() {
                for neighbor in positions
                    .into_iter()
                    .flatten()
                    .filter(|other| *other != position)
                {
                    neighbors[position].insert(neighbor);
                }
            }
        }
    }
    Ok(neighbors)
}

#[cfg(test)]
mod tests;
