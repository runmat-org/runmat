use std::collections::BTreeMap;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{quality::predicate::PredicateSign, MeshingCancellationSignal};

use crate::cdt::{
    topology::{build_delaunay_volume_topology_with_regions, orient_materialized_volume_nodes},
    DelaunayConstraintFacetSide, DelaunayConstraints, DelaunayTopologyErrorKind,
    DelaunayTopologyOptions, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshStage, DelaunayVolumeNode, DelaunayVolumeTopology,
};

/// Visibility proves a conforming star, not empty circumspheres. Only an
/// independently admitted star may replace the general point-set provider.
pub(super) fn try_build_constraint_star(
    region: &PersistentEntityId,
    constraints: &DelaunayConstraints,
    options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeMeshError> {
    let Some(topology) =
        build_visible_constraint_star(region, constraints, options.topology, cancellation)?
    else {
        return Ok(None);
    };
    match crate::cdt::insertion::validate_locally_delaunay_volume_topology(
        &topology,
        options,
        cancellation,
    ) {
        Ok(()) => Ok(Some(topology)),
        Err(failure) if failure.kind == crate::cdt::DelaunayInsertionErrorKind::InvalidTopology => {
            tracing::info!(reason = %failure, "visible constraint star rejected by physical Delaunay admission");
            Ok(None)
        }
        Err(failure) => Err(super::coordinate_topology_error(
            failure,
            "constraint-star admission",
        )),
    }
}

fn build_visible_constraint_star(
    region: &PersistentEntityId,
    constraints: &DelaunayConstraints,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeMeshError> {
    if constraints.nodes.is_empty() || constraints.facets.is_empty() {
        return Ok(None);
    }
    let seed_coordinates = super::centroid::canonical_average(
        constraints
            .nodes
            .iter()
            .map(|node| node.coordinates_m.canonical_value()),
    )
    .expect("nonempty constraint node inventory checked above");
    let seed = DelaunayVolumeNode::explicit(
        super::region_seed::region_seed_identity(region),
        seed_coordinates,
    );
    for (facet_index, facet) in constraints.facets.iter().enumerate() {
        let nodes = facet.vertex_indices.map(|index| {
            DelaunayVolumeNode::explicit(
                constraints.nodes[index as usize].identity,
                constraints.nodes[index as usize]
                    .coordinates_m
                    .canonical_value(),
            )
        });
        let side = orient_materialized_volume_nodes([nodes[0], nodes[1], nodes[2], seed])
            .map_err(|failure| invalid(format!("constraint-star predicate failed: {failure:?}")))?;
        let expected = if matches!(
            &facet.positive_side,
            DelaunayConstraintFacetSide::Region(side_region) if side_region == region
        ) {
            PredicateSign::Negative
        } else if matches!(
            &facet.negative_side,
            DelaunayConstraintFacetSide::Region(side_region) if side_region == region
        ) {
            PredicateSign::Positive
        } else {
            tracing::info!(
                facet_index,
                "constraint star rejected facet without component region"
            );
            return Ok(None);
        };
        if side != expected {
            tracing::info!(
                facet_index,
                ?side,
                ?expected,
                seed_x = seed.coordinates_m.canonical_value()[0],
                seed_y = seed.coordinates_m.canonical_value()[1],
                seed_z = seed.coordinates_m.canonical_value()[2],
                "constraint star rejected non-visible facet"
            );
            return Ok(None);
        }
    }
    let mut nodes = constraints.volume_nodes();
    nodes.push(seed);
    nodes.sort_by_key(|node| node.identity);
    let indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let seed_index = indices[&seed.identity];
    // Constraint recovery and carving own region assignment.  The star is an
    // initial conforming point-set topology, just like the generic Delaunay
    // path, so its tetrahedra deliberately remain unclassified here.
    let tetrahedra = constraints
        .facets
        .iter()
        .map(|facet| {
            (
                [
                    indices[&constraints.nodes[facet.vertex_indices[0] as usize].identity],
                    indices[&constraints.nodes[facet.vertex_indices[1] as usize].identity],
                    indices[&constraints.nodes[facet.vertex_indices[2] as usize].identity],
                    seed_index,
                ],
                None,
            )
        })
        .collect();
    match build_delaunay_volume_topology_with_regions(nodes, tetrahedra, options, cancellation) {
        Ok(topology) => Ok(Some(topology)),
        Err(failure) if failure.kind == DelaunayTopologyErrorKind::ResourceLimit => {
            Err(DelaunayVolumeMeshError {
                stage: DelaunayVolumeMeshStage::PointSet,
                kind: DelaunayVolumeMeshErrorKind::ResourceLimit,
                reason: failure.to_string(),
            })
        }
        Err(failure) if failure.kind == DelaunayTopologyErrorKind::Cancelled => {
            Err(DelaunayVolumeMeshError {
                stage: DelaunayVolumeMeshStage::PointSet,
                kind: DelaunayVolumeMeshErrorKind::Cancelled,
                reason: failure.to_string(),
            })
        }
        Err(failure) => {
            tracing::info!(reason = %failure, "constraint star topology rejected");
            Ok(None)
        }
    }
}

fn invalid(reason: impl Into<String>) -> DelaunayVolumeMeshError {
    DelaunayVolumeMeshError {
        stage: DelaunayVolumeMeshStage::PointSet,
        kind: DelaunayVolumeMeshErrorKind::InvalidTopology,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
