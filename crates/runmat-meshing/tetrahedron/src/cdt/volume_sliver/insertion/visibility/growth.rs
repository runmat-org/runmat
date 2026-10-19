//! Monotone cavity growth updates four faces per added cell. Fixed-point metric
//! scores are resolved once per boundary face, outside repeated growth decisions.
use super::*;
mod frontier;
use frontier::{Decision, Frontier};
#[cfg(test)]
mod tests;
pub(in crate::cdt::volume_sliver) fn grow_visibility_cavity(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    request: VisibilityCavityRequest<'_>,
) -> Result<VisibilityCavityResult, DelaunayVolumeSliverError> {
    let VisibilityCavityRequest {
        node,
        source_index,
        neighbor_index,
        relocated_vertices,
        protected_faces,
        quality_growth,
        options,
        cancellation,
    } = request;
    if !topology.contains_cell(source_index) || !topology.contains_cell(neighbor_index) {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver cavity source or neighbor is absent",
        ));
    }
    let source = topology.cell(source_index);
    let neighbor = topology.cell(neighbor_index);
    if source.region_id != neighbor.region_id || !source.neighbors.contains(&Some(neighbor_index)) {
        return Ok(Err(
            "sliver cavity tetrahedra are not same-region neighbors".into(),
        ));
    }
    let region = source.region_id.clone();
    let mut removed = BTreeSet::from([source_index, neighbor_index]);
    for vertex in relocated_vertices {
        removed.extend(topology.ordered_vertex_star(*vertex));
    }
    let check = |count: usize| {
        if count as u64 > options.insertion.maximum_cavity_tetrahedra {
            Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                "sliver insertion visibility cavity exceeded its tetrahedron limit",
            ))
        } else if cancellation.is_cancelled() {
            Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "sliver insertion visibility cavity was cancelled",
            ))
        } else {
            Ok(())
        }
    };
    check(removed.len())?;
    let mut frontier = Frontier::new(
        topology,
        node,
        protected_faces,
        region.clone(),
        quality_growth,
    );
    for (cell, opposite, face) in cavity_boundary_faces(topology, &removed) {
        frontier.insert(cell, opposite, face);
    }
    loop {
        check(removed.len())?;
        match frontier.next()? {
            Decision::Done => return Ok(Ok((removed, region))),
            Decision::Reject(reason) => return Ok(Err(reason.into())),
            Decision::Expand(adjacent) => {
                if !removed.insert(adjacent) {
                    return Ok(Err(
                        "sliver cavity visibility growth did not make progress".into()
                    ));
                }
                check(removed.len())?;
                let cell = topology.cell(adjacent);
                for opposite in 0..4 {
                    if let Some(neighbor) =
                        cell.neighbors[opposite].filter(|index| removed.contains(index))
                    {
                        let reciprocal = topology
                            .cell(neighbor)
                            .neighbors
                            .iter()
                            .position(|index| *index == Some(adjacent))
                            .ok_or_else(|| {
                                error(
                                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                                    "cavity growth lacks reciprocal neighbor authority",
                                )
                            })?;
                        frontier.remove(neighbor, reciprocal);
                    } else {
                        let mut face = [0; 3];
                        let mut write = 0;
                        for corner in 0..4 {
                            if corner != opposite {
                                face[write] = cell.vertex_indices[corner];
                                write += 1;
                            }
                        }
                        face.sort_unstable();
                        frontier.insert(adjacent, opposite, face);
                    }
                }
            }
        }
    }
}
