use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    checkpoint, error, validate_delaunay_volume_provenance, validate_options,
    DelaunayFacetProvenance, DelaunayNodeProvenance, DelaunaySegmentProvenance,
    DelaunayVolumeProvenance, DelaunayVolumeProvenanceError, DelaunayVolumeProvenanceErrorKind,
    DelaunayVolumeProvenanceOptions,
};
use crate::cdt::{
    validate_delaunay_carving, DelaunayCarving, DelaunayCarvingErrorKind, DelaunayCarvingOptions,
    DelaunayConstraintFacetSide, DelaunayConstraints, DelaunayFacetRecovery,
};

pub fn build_delaunay_volume_provenance(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    carving: &DelaunayCarving,
    carving_options: DelaunayCarvingOptions,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeProvenance, DelaunayVolumeProvenanceError> {
    validate_sources(
        recovery,
        constraints,
        carving,
        carving_options,
        options,
        cancellation,
    )?;
    let provenance = construct(recovery, constraints, carving, options, cancellation)?;
    validate_delaunay_volume_provenance(&carving.topology, &provenance, options, cancellation)?;
    Ok(provenance)
}

pub fn validate_delaunay_volume_provenance_sources(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    carving: &DelaunayCarving,
    provenance: &DelaunayVolumeProvenance,
    carving_options: DelaunayCarvingOptions,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeProvenanceError> {
    validate_sources(
        recovery,
        constraints,
        carving,
        carving_options,
        options,
        cancellation,
    )?;
    validate_delaunay_volume_provenance(&carving.topology, provenance, options, cancellation)?;
    let expected = construct(recovery, constraints, carving, options, cancellation)?;
    if provenance != &expected {
        return Err(error(
            DelaunayVolumeProvenanceErrorKind::InvalidProvenance,
            "volume provenance differs from canonical recovered exact-constraint lineage",
        ));
    }
    Ok(())
}

fn validate_sources(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    carving: &DelaunayCarving,
    carving_options: DelaunayCarvingOptions,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeProvenanceError> {
    validate_options(options)?;
    validate_delaunay_carving(
        recovery,
        constraints,
        carving,
        carving_options,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            DelaunayCarvingErrorKind::InvalidOptions => {
                DelaunayVolumeProvenanceErrorKind::InvalidOptions
            }
            DelaunayCarvingErrorKind::ResourceLimit => {
                DelaunayVolumeProvenanceErrorKind::ResourceLimit
            }
            DelaunayCarvingErrorKind::Cancelled => DelaunayVolumeProvenanceErrorKind::Cancelled,
            DelaunayCarvingErrorKind::InvalidTopology => {
                DelaunayVolumeProvenanceErrorKind::InvalidTopology
            }
            DelaunayCarvingErrorKind::InvalidConstraints
            | DelaunayCarvingErrorKind::AmbiguousClassification => {
                DelaunayVolumeProvenanceErrorKind::InvalidProvenance
            }
        };
        error(kind, failure.to_string())
    })
}

fn construct(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    carving: &DelaunayCarving,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeProvenance, DelaunayVolumeProvenanceError> {
    let retained_nodes = carving
        .topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    let mut nodes = Vec::new();
    for (index, node) in constraints.nodes.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        if let Some(vertex) = &node.source_vertex_id {
            if !retained_nodes.contains(&node.identity) {
                return Err(missing_retained_simplex(
                    "persistent exact vertex was removed during carving",
                ));
            }
            require_capacity(nodes.len(), options.maximum_node_bindings)?;
            nodes.push(DelaunayNodeProvenance {
                node_identity: node.identity,
                entity_ids: vec![vertex.clone()],
            });
        }
    }
    nodes.sort_by_key(|binding| binding.node_identity);

    let mut segments = Vec::new();
    for (index, recovered) in recovery.segment_recovery.segments.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let constraint = &constraints.segments[recovered.constraint_index as usize];
        let Some(source_edge) = &constraint.source_edge_id else {
            continue;
        };
        let source_parameters = constraint.source_edge_parameters.ok_or_else(|| {
            error(
                DelaunayVolumeProvenanceErrorKind::InvalidProvenance,
                "persistent exact edge support is missing its parameter interval",
            )
        })?;
        for pair in recovered.nodes.windows(2) {
            let mut identities = [pair[0].identity, pair[1].identity];
            let mut edge_parameters = [
                source_parameter(source_parameters, pair[0])?,
                source_parameter(source_parameters, pair[1])?,
            ];
            if identities[0] > identities[1] {
                edge_parameters.swap(0, 1);
            }
            identities.sort_unstable();
            if !identities
                .iter()
                .all(|identity| retained_nodes.contains(identity))
            {
                return Err(missing_retained_simplex(
                    "persistent exact edge support was removed during carving",
                ));
            }
            require_capacity(segments.len(), options.maximum_segment_bindings)?;
            segments.push(DelaunaySegmentProvenance {
                node_identities: identities,
                entity_ids: vec![source_edge.clone()],
                edge_parameters,
            });
        }
    }
    segments.sort_by_key(|binding| binding.node_identities);
    reject_duplicate_segments(&segments)?;

    let mut facets = Vec::new();
    for (index, recovered) in recovery.facets.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let constraint = &constraints.facets[recovered.constraint_index as usize];
        let mut entity_ids = Vec::with_capacity(1 + constraint.contact_ids.len());
        entity_ids.push(constraint.source_face_id.clone());
        entity_ids.extend(constraint.contact_ids.iter().cloned());
        entity_ids.sort();
        let region_ids = facet_regions(constraint);
        for triangle in &recovered.triangles {
            let mut identities = triangle.node_identities;
            identities.sort_unstable();
            if !identities
                .iter()
                .all(|identity| retained_nodes.contains(identity))
            {
                return Err(missing_retained_simplex(
                    "persistent exact face support was removed during carving",
                ));
            }
            require_capacity(facets.len(), options.maximum_facet_bindings)?;
            facets.push(DelaunayFacetProvenance {
                node_identities: identities,
                entity_ids: entity_ids.clone(),
                region_ids: region_ids.clone(),
            });
        }
    }
    facets.sort_by_key(|binding| binding.node_identities);
    reject_duplicate_facets(&facets)?;

    Ok(DelaunayVolumeProvenance {
        nodes,
        segments,
        facets,
    })
}

fn source_parameter(
    endpoints: [f64; 2],
    node: crate::cdt::DelaunayRecoveredSegmentNode,
) -> Result<f64, DelaunayVolumeProvenanceError> {
    let fraction = node.parameter().map_err(|failure| {
        error(
            DelaunayVolumeProvenanceErrorKind::InvalidProvenance,
            format!("recovered exact edge parameter is invalid: {failure}"),
        )
    })?;
    Ok(endpoints[0] * (1.0 - fraction) + endpoints[1] * fraction)
}

fn facet_regions(facet: &crate::cdt::DelaunayConstraintFacet) -> Vec<PersistentEntityId> {
    let mut regions = [&facet.positive_side, &facet.negative_side]
        .into_iter()
        .filter_map(|side| match side {
            DelaunayConstraintFacetSide::Region(region) => Some(region.clone()),
            DelaunayConstraintFacetSide::Exterior | DelaunayConstraintFacetSide::Void => None,
        })
        .collect::<Vec<_>>();
    regions.sort();
    regions
}

fn reject_duplicate_segments(
    segments: &[DelaunaySegmentProvenance],
) -> Result<(), DelaunayVolumeProvenanceError> {
    if segments
        .windows(2)
        .any(|pair| pair[0].node_identities == pair[1].node_identities)
    {
        return Err(error(
            DelaunayVolumeProvenanceErrorKind::InvalidProvenance,
            "distinct recovered exact edges claim one volume segment",
        ));
    }
    Ok(())
}

fn reject_duplicate_facets(
    facets: &[DelaunayFacetProvenance],
) -> Result<(), DelaunayVolumeProvenanceError> {
    if facets
        .windows(2)
        .any(|pair| pair[0].node_identities == pair[1].node_identities)
    {
        return Err(error(
            DelaunayVolumeProvenanceErrorKind::InvalidProvenance,
            "distinct recovered exact faces claim one volume facet",
        ));
    }
    Ok(())
}

fn missing_retained_simplex(reason: &'static str) -> DelaunayVolumeProvenanceError {
    error(DelaunayVolumeProvenanceErrorKind::InvalidProvenance, reason)
}

fn require_capacity(current: usize, maximum: u64) -> Result<(), DelaunayVolumeProvenanceError> {
    if current as u64 >= maximum {
        return Err(error(
            DelaunayVolumeProvenanceErrorKind::ResourceLimit,
            "constructed provenance inventory exceeds its hard limit",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod parameter_tests {
    use runmat_meshing_core::StableDigest;

    use super::source_parameter;

    #[test]
    fn dyadic_recovery_fraction_interpolates_the_exact_edge_interval() {
        let node = crate::cdt::DelaunayRecoveredSegmentNode {
            identity: StableDigest::from_bytes([9; 32]),
            parameter_numerator: 1,
            parameter_exponent: 2,
        };
        assert_eq!(source_parameter([2.0, 6.0], node).unwrap(), 3.0);
        assert_eq!(source_parameter([6.0, 2.0], node).unwrap(), 5.0);
    }
}
