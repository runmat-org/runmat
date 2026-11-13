//! Shared all-point empty-sphere queries for complete repair and cavity scoring.
//! Their global cost remains explicit; cycle and boundary hints do not replace them.
use super::*;

pub(super) fn first_global_delaunay_violation(
    topology: &DelaunayVolumeTopology,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(usize, usize)>, DelaunayInsertionError> {
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        let points = tetrahedron
            .vertex_indices
            .map(|node| topology.nodes[node as usize]);
        for (node_index, node) in topology.nodes.iter().copied().enumerate() {
            if tetrahedron.vertex_indices.contains(&(node_index as u32))
                || topology.incidence.vertex_stars[node_index].is_empty()
            {
                continue;
            }
            work.predicate()?;
            if insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                return Ok(Some((tetrahedron_index, node_index)));
            }
        }
    }
    Ok(None)
}

pub(super) fn global_delaunay_violation_queries(
    topology: &DelaunayVolumeTopology,
    work: &mut InsertionWork<'_>,
) -> Result<Vec<usize>, DelaunayInsertionError> {
    let mut queries = BTreeSet::new();
    for tetrahedron in &topology.tetrahedra {
        let points = tetrahedron
            .vertex_indices
            .map(|node| topology.nodes[node as usize]);
        for (node_index, node) in topology.nodes.iter().copied().enumerate() {
            if tetrahedron.vertex_indices.contains(&(node_index as u32))
                || topology.incidence.vertex_stars[node_index].is_empty()
                || queries.contains(&node_index)
            {
                continue;
            }
            work.predicate()?;
            if insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                queries.insert(node_index);
            }
        }
    }
    Ok(queries.into_iter().collect())
}

pub(super) fn indexed_tetrahedra_global_violation_count(
    topology: &DelaunayVolumeTopology,
    tetrahedra: &[usize],
    work: &mut InsertionWork<'_>,
) -> Result<u64, DelaunayInsertionError> {
    tetrahedra_global_violation_count(
        topology,
        &tetrahedra
            .iter()
            .map(|index| topology.tetrahedra[*index].vertex_indices)
            .collect::<Vec<_>>(),
        work,
    )
}

pub(super) fn tetrahedra_global_violation_count(
    topology: &DelaunayVolumeTopology,
    tetrahedra: &[[u32; 4]],
    work: &mut InsertionWork<'_>,
) -> Result<u64, DelaunayInsertionError> {
    let mut violations = 0_u64;
    for tetrahedron in tetrahedra {
        let mut tetrahedron = *tetrahedron;
        match orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
            .map_err(super::predicate_error)?
        {
            PredicateSign::Positive => {}
            PredicateSign::Negative => tetrahedron.swap(0, 1),
            PredicateSign::Zero => return Ok(u64::MAX),
        }
        let points = tetrahedron.map(|node| topology.nodes[node as usize]);
        for (node_index, node) in topology.nodes.iter().copied().enumerate() {
            if tetrahedron.contains(&(node_index as u32)) {
                continue;
            }
            work.predicate()?;
            if insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
                .map_err(super::predicate_error)?
                == PredicateSign::Positive
            {
                violations = violations.checked_add(1).ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::ResourceLimit,
                        "global Delaunay violation count overflow",
                    )
                })?;
            }
        }
    }
    Ok(violations)
}
