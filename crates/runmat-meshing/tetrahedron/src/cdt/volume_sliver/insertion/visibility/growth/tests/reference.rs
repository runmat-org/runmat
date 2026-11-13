//! Independent retained scan-based reference for small differential tests.
use super::*;
pub(super) fn grow(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    request: VisibilityCavityRequest<'_>,
) -> Result<VisibilityCavityResult, DelaunayVolumeSliverError> {
    let VisibilityCavityRequest {
        node,
        source_index,
        relocated_vertices,
        neighbor_index,
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
            "sliver cavity tetrahedra are not same-region neighbors".to_owned(),
        ));
    }
    let region = source.region_id.clone();
    let mut removed = BTreeSet::from([source_index, neighbor_index]);
    for vertex in relocated_vertices {
        removed.extend(topology.ordered_vertex_star(*vertex));
    }

    loop {
        if removed.len() as u64 > options.insertion.maximum_cavity_tetrahedra {
            return Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                "sliver insertion visibility cavity exceeded its tetrahedron limit",
            ));
        }
        if cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                "sliver insertion visibility cavity was cancelled",
            ));
        }
        let mut expansion = None;
        'cavity: for (tetrahedron_index, opposite, face) in
            cavity_boundary_faces(topology, &removed)
        {
            let tetrahedron = &topology.cell(tetrahedron_index);
            let face_nodes = face.map(|vertex| topology.node(vertex));
            let interior = topology.node(tetrahedron.vertex_indices[opposite]);
            let interior_side =
                orient_volume_nodes([face_nodes[0], face_nodes[1], face_nodes[2], interior])
                    .map_err(visibility_predicate_error)?;
            let point_side =
                orient_volume_nodes([face_nodes[0], face_nodes[1], face_nodes[2], node])
                    .map_err(visibility_predicate_error)?;
            if point_side != PredicateSign::Zero && point_side == interior_side {
                continue;
            }
            let face_identities = face.map(|vertex| topology.node(vertex).identity);
            if protected_faces.contains(&face_identities) {
                return Ok(Err(
                    "sliver cavity visibility growth reached a protected facet".to_owned(),
                ));
            }
            let Some(adjacent) = tetrahedron.neighbors[opposite] else {
                return Ok(Err(
                    "sliver insertion point lies outside the volume boundary".to_owned(),
                ));
            };
            if topology.cell(adjacent).region_id != &region {
                return Ok(Err(
                    "sliver cavity visibility growth reached a region boundary".to_owned(),
                ));
            }
            expansion = Some(adjacent);
            break 'cavity;
        }
        let adjacent = match expansion {
            Some(adjacent) => adjacent,
            None => {
                let Some((metrics, target)) = quality_growth else {
                    return Ok(Ok((removed, region)));
                };
                let mut worst: Option<(f64, u32, usize, [u32; 3])> = None;
                for (tetrahedron_index, opposite, face) in cavity_boundary_faces(topology, &removed)
                {
                    let tetrahedron = topology.cell(tetrahedron_index);
                    let Some(adjacent) = tetrahedron.neighbors[opposite] else {
                        continue;
                    };
                    let face_identities = face.map(|vertex| topology.node(vertex).identity);
                    // A blocked face is not an expansion candidate. It must not
                    // hide another poor, expandable face on this same cavity.
                    if protected_faces.contains(&face_identities)
                        || topology.cell(adjacent).region_id != &region
                    {
                        continue;
                    }
                    let points = [
                        topology.node(face[0]).coordinates_m.canonical_value(),
                        topology.node(face[1]).coordinates_m.canonical_value(),
                        topology.node(face[2]).coordinates_m.canonical_value(),
                        node.coordinates_m.canonical_value(),
                    ];
                    let score = crate::cdt::volume_quality::evaluate_tetrahedron_jacobian(
                        points,
                        metrics(face)?,
                    )
                    .unwrap()
                    .unwrap_or(0.);
                    if worst.as_ref().is_none_or(|previous| {
                        score
                            .total_cmp(&previous.0)
                            .then_with(|| {
                                face.map(|vertex| topology.node(vertex).identity)
                                    .cmp(&previous.3.map(|vertex| topology.node(vertex).identity))
                            })
                            .is_lt()
                    }) {
                        worst = Some((score, tetrahedron_index, opposite, face));
                    }
                }
                let Some((score, tetrahedron_index, opposite, _face)) = worst else {
                    return Ok(Ok((removed, region)));
                };
                if score > target {
                    return Ok(Ok((removed, region)));
                }
                topology.cell(tetrahedron_index).neighbors[opposite]
                    .expect("quality expansion candidate retains its adjacent cell")
            }
        };
        if !removed.insert(adjacent) {
            return Ok(Err(
                "sliver cavity visibility growth did not make progress".to_owned()
            ));
        }
    }
}
