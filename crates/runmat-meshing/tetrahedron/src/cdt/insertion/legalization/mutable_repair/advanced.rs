//! Advanced physical repair stays in changed neighborhoods of an admitted prefix.
//! Shared cavity geometry/lifting rules and checked journals remain authoritative.
//! Failure retains dirty evidence and takes the complete independent fallback.
use super::*;
#[cfg(test)]
mod tests;

type Recipe = (Vec<usize>, Vec<[u32; 4]>);

pub(super) fn violations(
    topology: &TopologyMutation,
    work: &mut InsertionWork<'_>,
) -> Result<(u64, BTreeSet<u32>), DelaunayInsertionError> {
    let mut count = 0;
    let mut queries = BTreeSet::new();
    for shared in topology
        .physical_dirty_faces()
        .expect("admitted physical prefix")
    {
        work.checkpoint()?;
        let Some(sides) = violation(topology, *shared, &BTreeSet::new(), work)? else {
            continue;
        };
        count += 1;
        for (index, opposite) in sides {
            queries.insert(
                topology
                    .tetrahedron(index)
                    .expect("live dirty face")
                    .vertex_indices[opposite],
            );
        }
    }
    Ok((count, queries))
}

pub(super) fn repair(
    topology: &mut TopologyMutation,
    before: u64,
    queries: &BTreeSet<u32>,
    visited: &BTreeSet<[u8; 32]>,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    let barriers = BTreeSet::new();
    let mut edges = BTreeSet::new();
    for query in queries {
        for index in topology.vertex_star(*query) {
            work.checkpoint()?;
            let vertices = topology
                .tetrahedron(index)
                .expect("live query star")
                .vertex_indices;
            for left in 0..4 {
                for right in left + 1..4 {
                    let mut edge = [vertices[left], vertices[right]];
                    edge.sort_unstable();
                    edges.insert(edge);
                }
            }
        }
    }
    let mut plateau = None;
    for edge in edges {
        let removed = topology
            .edge_tetrahedra(edge)
            .into_iter()
            .map(|index| index as usize)
            .collect::<Vec<_>>();
        for _ in &removed {
            work.checkpoint()?;
        }
        if removed.len() < 3 {
            continue;
        }
        let region = &topology
            .tetrahedron(removed[0] as u32)
            .expect("live star")
            .region_id;
        if removed.iter().any(|index| {
            &topology
                .tetrahedron(*index as u32)
                .expect("live star")
                .region_id
                != region
        }) {
            continue;
        }
        let boundary = local_tetrahedron_boundary_faces(
            &removed
                .iter()
                .map(|index| {
                    topology
                        .tetrahedron(*index as u32)
                        .expect("live star")
                        .vertex_indices
                })
                .collect::<Vec<_>>(),
        );
        let candidates = crate::reconnect::BoundaryEdgeStarCandidates::new(&boundary, &mut || {
            work.checkpoint()
        })?;
        for target in candidates.targets() {
            work.checkpoint()?;
            if target == edge || super::super::topology_has_edge(topology, target) {
                continue;
            }
            let Some(created) =
                super::super::edge_star_replacement(topology, &candidates, target, work)?
            else {
                continue;
            };
            if consider(
                topology,
                (&removed, &created),
                before,
                visited,
                &mut plateau,
                work,
            )? {
                return Ok(true);
            }
        }
    }
    if let Some(recipe) = plateau {
        return accept_recipe(topology, &recipe, work);
    }
    for query in queries {
        let Some((removed, created)) =
            super::super::vertex_reinsertion_geometry(topology, *query as usize, &barriers, work)?
        else {
            continue;
        };
        if consider(
            topology,
            (&removed, &created),
            before,
            visited,
            &mut plateau,
            work,
        )? {
            return Ok(true);
        }
    }
    if let Some(recipe) = plateau {
        return accept_recipe(topology, &recipe, work);
    }
    Ok(false)
}

fn consider(
    topology: &mut TopologyMutation,
    recipe: (&[usize], &[[u32; 4]]),
    before: u64,
    visited: &BTreeSet<[u8; 32]>,
    plateau: &mut Option<Recipe>,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    let (removed, created) = recipe;
    if !super::super::tetrahedra_are_materializable(topology, created)? {
        return Ok(false);
    }
    let change = super::super::progress::lifting_change(topology, removed, created, work)?;
    if change == PredicateSign::Positive || change == PredicateSign::Zero && plateau.is_some() {
        return Ok(false);
    }
    let mut trial = topology.cavity_trial();
    if !apply(&mut trial, &(removed.to_vec(), created.to_vec()), work)? {
        return Ok(false);
    }
    let fingerprint = trial.connectivity_search_fingerprint(&mut || work.checkpoint())?;
    if visited.contains(&fingerprint) {
        return Ok(false);
    }
    if change == PredicateSign::Negative {
        record_flip(work)?;
        trial.commit();
        return Ok(true);
    }
    if violations(&trial, work)?.0 <= before {
        *plateau = Some((removed.to_vec(), created.to_vec()));
    }
    Ok(false)
}

fn apply(
    topology: &mut TopologyMutation,
    (removed, created): &Recipe,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    let count = topology
        .active_tetrahedron_count()
        .checked_sub(removed.len())
        .and_then(|count| count.checked_add(created.len()))
        .ok_or_else(|| super::super::super::resource("advanced repair cell count overflow"))?;
    if count as u64 > work.options.topology.maximum_tetrahedra {
        return Err(super::super::super::resource(
            "advanced repair cell inventory limit exceeded",
        ));
    }
    let region = topology
        .tetrahedron(removed[0] as u32)
        .expect("live cavity")
        .region_id
        .clone();
    match topology.replace_cavity(
        &removed.iter().map(|index| *index as u32).collect(),
        created.iter().map(|cell| (*cell, region.clone())).collect(),
    ) {
        Ok(_) => Ok(true),
        Err(failure) if failure.kind == TopologyMutationErrorKind::InvalidTopology => Ok(false),
        Err(failure) => Err(super::super::super::mutation_topology_error(failure)),
    }
}

fn accept_recipe(
    topology: &mut TopologyMutation,
    recipe: &Recipe,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    let mut trial = topology.cavity_trial();
    if !apply(&mut trial, recipe, work)? {
        return Ok(false);
    }
    record_flip(work)?;
    trial.commit();
    Ok(true)
}
