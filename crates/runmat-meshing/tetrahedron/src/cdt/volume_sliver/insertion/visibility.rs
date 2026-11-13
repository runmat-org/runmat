//! Exact visibility-cavity growth, smoothing and conforming insertion.
use super::*;
#[cfg(test)]
use crate::cdt::insertion::topology_access::TetrahedralTopology;
mod kernel;
mod smoothing;
pub(in crate::cdt::volume_sliver) use smoothing::optimize_cavity_point;
#[cfg(test)]
mod tests;

pub(in crate::cdt::volume_sliver) use growth::grow_visibility_cavity;
mod growth;

pub(in crate::cdt::volume_sliver) fn cavity_boundary_faces(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    removed: &BTreeSet<u32>,
) -> Vec<(u32, usize, [u32; 3])> {
    let mut faces = Vec::new();
    for tetrahedron_index in removed.iter().copied() {
        let tetrahedron = &topology.cell(tetrahedron_index);
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite].is_some_and(|adjacent| removed.contains(&adjacent)) {
                continue;
            }
            let mut face = [0_u32; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.into_iter().enumerate() {
                if slot != opposite {
                    face[cursor] = vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable_by_key(|vertex| topology.node(*vertex).identity);
            faces.push((tetrahedron_index, opposite, face));
        }
    }
    faces.sort_by_key(|(_, _, face)| face.map(|vertex| topology.node(vertex).identity));
    faces
}

pub(super) fn visibility_predicate_error(
    failure: runmat_meshing_core::quality::predicate::SpatialPredicateError,
) -> DelaunayVolumeSliverError {
    error(
        DelaunayVolumeSliverErrorKind::InvalidTopology,
        format!("sliver cavity visibility predicate failed: {failure:?}"),
    )
}

pub(super) fn smooth_visibility_cavity_point(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    start: [f64; 3],
    context: VisibilityContext<'_>,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let VisibilityContext {
        source_vertices,
        neighbor_index,
        protected_faces,
        metrics,
        target_scaled_jacobian,
        options,
        cancellation,
    } = context;
    let source_index = topology
        .tetrahedron_index(source_vertices)
        .expect("quality source remains in topology");
    let probe = |point| DelaunayVolumeNode::explicit(kernel_probe_identity(point), point);
    let Ok((removed, _)) = grow_visibility_cavity(
        topology,
        VisibilityCavityRequest {
            relocated_vertices: &[],
            node: probe(start),
            source_index,
            neighbor_index,
            protected_faces,
            quality_growth: Some((metrics, target_scaled_jacobian)),
            options,
            cancellation,
        },
    )?
    else {
        return Ok(None);
    };
    optimize_cavity_point(topology, &removed, start, metrics, cancellation)
}

#[cfg(test)]
pub(super) fn split_two_tetrahedron_cavity(
    topology: &DelaunayVolumeTopology,
    node: DelaunayVolumeNode,
    context: VisibilityContext<'_>,
    work: &mut OptimizationWork<'_>,
) -> Result<EdgeStarSplit, DelaunayVolumeSliverError> {
    let VisibilityContext {
        source_vertices,
        neighbor_index,
        protected_faces,
        metrics,
        target_scaled_jacobian,
        options,
        cancellation,
    } = context;
    let Some(source_index) = topology
        .tetrahedra
        .iter()
        .position(|tetrahedron| {
            source_vertices
                .iter()
                .all(|vertex| tetrahedron.vertex_indices.contains(vertex))
        })
        .and_then(|index| u32::try_from(index).ok())
    else {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver cavity source tetrahedron is absent",
        ));
    };
    let (removed, region) = match grow_visibility_cavity(
        topology,
        VisibilityCavityRequest {
            relocated_vertices: &[],
            node,
            source_index,
            neighbor_index,
            protected_faces,
            quality_growth: Some((metrics, target_scaled_jacobian)),
            options,
            cancellation,
        },
    )? {
        Ok(cavity) => cavity,
        Err(detail) => return Ok(EdgeStarSplit::Rejected(detail)),
    };

    let removed_tetrahedra = removed
        .iter()
        .map(|index| topology.cell(*index).vertex_indices)
        .collect::<Vec<_>>();
    let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra)
        .into_iter()
        .collect::<Vec<_>>();

    let created = match super::refill::preserve_vertices(
        topology,
        &removed,
        &boundary,
        super::refill::RefillPoint::insert(node),
        options,
        work,
        cancellation,
    ) {
        Ok(created) => created,
        Err(failure) if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology => {
            return Ok(EdgeStarSplit::Rejected(failure.to_string()))
        }
        Err(failure) => return Err(failure),
    };
    let mut mutation = TopologyMutation::from_canonical(topology.clone());
    if let Err(failure) = mutation.append_node(node) {
        return Ok(EdgeStarSplit::Rejected(failure.to_string()));
    }
    let created = created
        .into_iter()
        .map(|identities| {
            (
                identities.map(|identity| {
                    mutation
                        .node_index(identity)
                        .expect("preserved cavity point")
                }),
                region.clone(),
            )
        })
        .collect();
    if let Err(failure) = mutation.replace_cavity(&removed, created) {
        return Ok(EdgeStarSplit::Rejected(failure.to_string()));
    }
    match mutation.freeze_canonical(options.insertion.topology, cancellation) {
        Ok(topology) => Ok(EdgeStarSplit::Topology(topology)),
        Err(failure) => match failure.kind {
            crate::cdt::mutation::TopologyMutationErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            crate::cdt::mutation::TopologyMutationErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology => {
                Ok(EdgeStarSplit::Rejected(failure.to_string()))
            }
        },
    }
}
