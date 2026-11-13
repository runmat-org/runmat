//! Plan only the changed closed cavity; the working owner checks and applies it.
use super::*;

pub(super) fn split(
    trial: &mut WorkingTrial<'_, '_>,
    node: DelaunayVolumeNode,
    choice: EdgeStarChoice,
    source: &DelaunayTetrahedronQuality,
    policy: SliverSearchPolicy,
    work: &mut OptimizationWork<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Result<(), String>, DelaunayVolumeSliverError> {
    let topology = trial.topology();
    let vertices = source.node_identities.map(|identity| {
        topology
            .node_index(identity)
            .expect("quality source corner")
    });
    let cavity = match choice.strategy {
        InsertionStrategy::Cavity { neighbor_index }
        | InsertionStrategy::BoundaryOffcenter { neighbor_index } => {
            visibility::grow_visibility_cavity(
                topology,
                VisibilityCavityRequest {
                    relocated_vertices: &[],
                    node,
                    source_index: topology
                        .tetrahedron_index(vertices)
                        .expect("quality source"),
                    neighbor_index,
                    protected_faces: trial.preparation().protected_facets(),
                    quality_growth: Some((
                        &|face: [u32; 3]| {
                            trial
                                .preparation()
                                .insertion_metric(
                                    face.map(|vertex| topology.node(vertex).identity),
                                    &source.region_id,
                                )
                                .map_err(crate::cdt::volume_sliver::quality_error)
                        },
                        trial.quality_options().minimum_metric_scaled_jacobian,
                    )),
                    options: policy.options,
                    cancellation,
                },
            )?
        }
        InsertionStrategy::EdgeStar => {
            let edge = choice
                .edge_identities
                .map(|identity| topology.node_index(identity).expect("choice edge"));
            let star = topology.edge_star(edge);
            if star.len() < 3 || topology.edge_is_boundary(edge) {
                return Ok(Err(
                    "selected edge does not have a complete interior star".to_owned()
                ));
            }
            let region = topology.cell(star[0]).region_id.clone();
            if star
                .iter()
                .any(|index| topology.cell(*index).region_id != &region)
            {
                return Ok(Err(
                    "selected edge star crosses a region boundary".to_owned()
                ));
            }
            Ok((star.into_iter().collect(), region))
        }
    };
    let (removed, region) = match cavity {
        Ok(cavity) => cavity,
        Err(detail) => return Ok(Err(detail)),
    };
    let boundary = visibility::cavity_boundary_faces(topology, &removed)
        .into_iter()
        .map(|(_, _, face)| face)
        .collect::<Vec<_>>();
    let created = match super::super::refill::preserve_vertices(
        topology,
        &removed,
        &boundary,
        super::super::refill::RefillPoint::insert(node),
        policy.options,
        work,
        cancellation,
    ) {
        Ok(created) => created,
        Err(failure) if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology => {
            return Ok(Err(failure.to_string()))
        }
        Err(failure) => return Err(failure),
    };
    trial.append_node(node, policy.options)?;
    let created = created
        .into_iter()
        .map(|identities| {
            (
                identities.map(|identity| {
                    trial
                        .topology()
                        .node_index(identity)
                        .expect("preserved cavity point")
                }),
                region.clone(),
            )
        })
        .collect();
    match trial
        .replace(&removed, created, policy.options, cancellation)
        .map_err(DelaunayVolumeSliverError::from)
    {
        Ok(()) => Ok(Ok(())),
        Err(failure) if failure.kind == DelaunayVolumeSliverErrorKind::InvalidTopology => {
            Ok(Err(failure.to_string()))
        }
        Err(failure) => Err(failure),
    }
}
