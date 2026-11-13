//! Exact local empty-sphere admission for a coverage-preserving cavity replacement.
//! The caller proves cavity coverage and preserves barriers. This checks every
//! changed interior and exterior adjacency; terminal whole-topology validation
//! remains mandatory and detects any pre-existing violation outside the cavity.
use super::topology_access::TetrahedralTopology;
use runmat_meshing_core::quality::predicate::{PredicateSign, SpatialPredicateError};
use std::collections::BTreeSet;

pub(in crate::cdt) enum LocalAdmissionError<E> {
    Work(E),
    Predicate(SpatialPredicateError),
}

pub(in crate::cdt) fn admits_replacement<E>(
    topology: &impl TetrahedralTopology,
    removed: &BTreeSet<u32>,
    created: &[[u32; 4]],
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<bool, LocalAdmissionError<E>> {
    let vertices = created.iter().flatten().copied().collect::<BTreeSet<_>>();
    for tetrahedron in created {
        let mut tetrahedron = *tetrahedron;
        checkpoint().map_err(LocalAdmissionError::Work)?;
        match topology
            .orient_nodes(tetrahedron.map(|v| topology.nodes()[v as usize]))
            .map_err(LocalAdmissionError::Predicate)?
        {
            PredicateSign::Zero => return Ok(false),
            PredicateSign::Negative => tetrahedron.swap(0, 1),
            PredicateSign::Positive => {}
        }
        // All replacement vertices are checked, including those beyond an
        // immediate new neighbor. The cavity inventory is bounded by its owner.
        for query in vertices
            .iter()
            .copied()
            .filter(|v| !tetrahedron.contains(v))
        {
            if inside(topology, tetrahedron, query, checkpoint)? {
                return Ok(false);
            }
        }
        for opposite in 0..4 {
            let face = tetrahedron
                .iter()
                .enumerate()
                .filter_map(|(slot, vertex)| (slot != opposite).then_some(*vertex))
                .collect::<Vec<_>>();
            for neighbor_index in topology
                .face_tetrahedra_with_checkpoint([face[0], face[1], face[2]], checkpoint)
                .map_err(LocalAdmissionError::Work)?
            {
                if removed.contains(&neighbor_index) {
                    continue;
                }
                let Some(neighbor) = topology.tetrahedron(neighbor_index) else {
                    continue;
                };
                if !face
                    .iter()
                    .all(|vertex| neighbor.vertex_indices.contains(vertex))
                {
                    continue;
                }
                let query = neighbor
                    .vertex_indices
                    .into_iter()
                    .find(|vertex| !face.contains(vertex))
                    .expect("tetrahedral face has one apex");
                if inside(topology, tetrahedron, query, checkpoint)?
                    || inside(
                        topology,
                        neighbor.vertex_indices,
                        tetrahedron[opposite],
                        checkpoint,
                    )?
                {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn inside<E>(
    topology: &impl TetrahedralTopology,
    tetrahedron: [u32; 4],
    query: u32,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<bool, LocalAdmissionError<E>> {
    checkpoint().map_err(LocalAdmissionError::Work)?;
    let mut tetrahedron = tetrahedron;
    match topology
        .orient_nodes(tetrahedron.map(|vertex| topology.nodes()[vertex as usize]))
        .map_err(LocalAdmissionError::Predicate)?
    {
        PredicateSign::Zero => return Ok(true),
        PredicateSign::Negative => tetrahedron.swap(0, 1),
        PredicateSign::Positive => {}
    }
    checkpoint().map_err(LocalAdmissionError::Work)?;
    let [first, second, third, fourth] =
        tetrahedron.map(|vertex| topology.nodes()[vertex as usize]);
    topology
        .insphere_nodes([
            first,
            second,
            third,
            fourth,
            topology.nodes()[query as usize],
        ])
        .map(|sign| sign == PredicateSign::Positive)
        .map_err(LocalAdmissionError::Predicate)
}
