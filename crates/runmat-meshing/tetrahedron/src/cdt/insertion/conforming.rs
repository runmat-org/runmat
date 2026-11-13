//! Explicit conforming insertion preserves cells outside the point's support.
//! It neither expands an Euclidean conflict sphere nor imposes Delaunay flips.
//! Quality remeshing can therefore retain its independently chosen geometry.
use super::*;
use std::collections::VecDeque;
#[cfg(test)]
mod tests;

pub(in crate::cdt) fn insert_conforming_volume_node_into_mutation_with_barriers(
    topology: &mut TopologyMutation,
    node: DelaunayVolumeNode,
    protected_faces: &BTreeSet<[runmat_meshing_core::StableDigest; 3]>,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_new_node(topology, node)?;
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(resource("protected-face inventory limit exceeded"));
    }
    let mut work = InsertionWork::new(options, cancellation);
    let seed = TetrahedronPointLocator::walking(topology)
        .first_containing_from(node, None, &mut work)?
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::PointOutsideTopology,
                "conforming insertion point is outside the admitted complex",
            )
        })?;
    let region = topology
        .tetrahedron(seed)
        .expect("located cell")
        .region_id
        .clone();
    let mut removed = BTreeSet::new();
    let mut queue = VecDeque::from([seed]);
    while let Some(index) = queue.pop_front() {
        work.checkpoint()?;
        if !removed.insert(index) {
            continue;
        }
        if removed.len() as u64 > options.maximum_cavity_tetrahedra {
            return Err(resource(
                "conforming insertion support exceeds its cavity limit",
            ));
        }
        let cell = topology.tetrahedron(index).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "conforming insertion support is inactive",
            )
        })?;
        if cell.region_id != region {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "conforming insertion crosses assigned region incidence",
            ));
        }
        for opposite in 0..4 {
            if !node_lies_on_face(topology, cell.vertex_indices, opposite, node, &mut work)? {
                continue;
            }
            let face = cavity::stable_face(topology, cell.vertex_indices, opposite);
            if protected_faces.contains(&face) {
                return Err(error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "conforming insertion would split a protected facet",
                ));
            }
            if let Some(adjacent) = cell.neighbors[opposite] {
                if !removed.contains(&adjacent) {
                    queue.push_back(adjacent);
                }
            }
        }
    }
    if !cavity_is_star_shaped(topology, &removed, node, &mut work)? {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "conforming insertion support is not star-shaped",
        ));
    }
    let boundary = cavity_boundary(topology, &removed, node, &mut work)?;
    mutable::require_mutation_capacity(topology, removed.len(), boundary.len(), options)?;
    let mut trial = topology.cavity_trial();
    let inserted = trial.append_node(node).map_err(mutation_topology_error)?;
    let created = boundary
        .into_iter()
        .map(|face| ([face[0], face[1], face[2], inserted], region.clone()))
        .collect();
    trial
        .replace_closed_cavity(
            &removed,
            created,
            protected_faces,
            options.topology,
            cancellation,
        )
        .map_err(mutation_topology_error)?;
    work.checkpoint()?;
    trial.commit();
    Ok(())
}
