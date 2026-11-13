use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    error, validate_input, validate_options, DelaunayVolumeSliverError,
    DelaunayVolumeSliverErrorKind, DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions,
    DelaunayVolumeSliverTreatment,
};
use crate::cdt::DelaunayVolumeRefinementInput;

pub fn validate_delaunay_volume_sliver_treatment(
    input: DelaunayVolumeRefinementInput<'_>,
    treatment: &DelaunayVolumeSliverTreatment,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeSliverError> {
    validate_options(options)?;
    validate_input(input, options, cancellation)?;
    if treatment.mutations.len() as u64 > options.maximum_passes
        || treatment.mutations.iter().any(|mutation| match mutation {
            DelaunayVolumeSliverMutation::Relocated(relocation) => {
                relocation.source_node_identity == StableDigest::ZERO
                    || relocation.replacement_node.identity == StableDigest::ZERO
                    || relocation.source_node_identity == relocation.replacement_node.identity
                    || relocation
                        .replacement_node
                        .coordinates_m
                        .canonical_value()
                        .iter()
                        .any(|value| !value.is_finite())
            }
            DelaunayVolumeSliverMutation::Deleted(deletion) => {
                deletion.source_node_identity == StableDigest::ZERO
                    || !deletion
                        .source_tetrahedron_node_identities
                        .contains(&deletion.source_node_identity)
            }
            DelaunayVolumeSliverMutation::Reconnected(reconnection) => {
                !valid_reconnection(reconnection)
            }
            DelaunayVolumeSliverMutation::Inserted(insertion) => {
                insertion.inserted_node.identity == StableDigest::ZERO
                    || insertion
                        .inserted_node
                        .coordinates_m
                        .canonical_value()
                        .iter()
                        .any(|value| !value.is_finite())
                    || !insertion
                        .source_tetrahedron_node_identities
                        .iter()
                        .all(|identity| *identity != StableDigest::ZERO)
                    || !insertion
                        .source_edge_node_identities
                        .iter()
                        .all(|identity| {
                            insertion
                                .source_tetrahedron_node_identities
                                .contains(identity)
                        })
                    || !super::insertion::insertion_identity_is_valid(insertion)
            }
        })
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver mutation evidence is malformed or exceeds its pass limit",
        ));
    }
    let replacement_identities = treatment
        .mutations
        .iter()
        .filter_map(|mutation| match mutation {
            DelaunayVolumeSliverMutation::Relocated(relocation) => {
                Some(relocation.replacement_node.identity)
            }
            DelaunayVolumeSliverMutation::Deleted(_) => None,
            DelaunayVolumeSliverMutation::Reconnected(_) => None,
            DelaunayVolumeSliverMutation::Inserted(_) => None,
        })
        .collect::<BTreeSet<_>>();
    let relocation_count = treatment
        .mutations
        .iter()
        .filter(|mutation| matches!(mutation, DelaunayVolumeSliverMutation::Relocated(_)))
        .count();
    if replacement_identities.len() != relocation_count {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "relocation replacement identities must be unique",
        ));
    }
    validate_mutation_lineage(input, treatment)?;
    if treatment
        .quality
        .tetrahedra
        .iter()
        .any(|tetrahedron| tetrahedron.requires_optimization())
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidQuality,
            "completed volume optimization retains a shape-quality violation",
        ));
    }
    validate_input(
        DelaunayVolumeRefinementInput {
            topology: &treatment.topology,
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality: &treatment.quality,
            quality_options: input.quality_options,
            fixed_node_identities: input.fixed_node_identities,
        },
        options,
        cancellation,
    )
}

fn validate_mutation_lineage(
    input: DelaunayVolumeRefinementInput<'_>,
    treatment: &DelaunayVolumeSliverTreatment,
) -> Result<(), DelaunayVolumeSliverError> {
    let mut nodes = input
        .topology
        .nodes
        .iter()
        .map(|node| (node.identity, *node))
        .collect::<BTreeMap<_, _>>();
    for mutation in &treatment.mutations {
        let valid = match mutation {
            DelaunayVolumeSliverMutation::Relocated(relocation) => {
                relocation
                    .source_tetrahedron_node_identities
                    .iter()
                    .all(|identity| nodes.contains_key(identity))
                    && super::relocation::relocation_identity_is_valid(relocation)
                    && nodes.remove(&relocation.source_node_identity).is_some()
                    && nodes
                        .insert(
                            relocation.replacement_node.identity,
                            relocation.replacement_node,
                        )
                        .is_none()
            }
            DelaunayVolumeSliverMutation::Deleted(deletion) => {
                deletion
                    .source_tetrahedron_node_identities
                    .iter()
                    .all(|identity| nodes.contains_key(identity))
                    && nodes.remove(&deletion.source_node_identity).is_some()
            }
            DelaunayVolumeSliverMutation::Reconnected(reconnection) => reconnection
                .source_tetrahedron_node_identities
                .iter()
                .chain(
                    reconnection
                        .removed_tetrahedron_node_identities
                        .iter()
                        .flatten(),
                )
                .chain(
                    reconnection
                        .created_tetrahedron_node_identities
                        .iter()
                        .flatten(),
                )
                .all(|identity| nodes.contains_key(identity)),
            DelaunayVolumeSliverMutation::Inserted(insertion) => {
                insertion
                    .source_tetrahedron_node_identities
                    .iter()
                    .chain(&insertion.source_edge_node_identities)
                    .all(|identity| nodes.contains_key(identity))
                    && nodes
                        .insert(insertion.inserted_node.identity, insertion.inserted_node)
                        .is_none()
            }
        };
        if !valid {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "sliver mutation lineage is inconsistent with its input nodes",
            ));
        }
    }
    if nodes.values().copied().collect::<Vec<_>>() != treatment.topology.nodes {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver mutation lineage does not produce the result node inventory",
        ));
    }
    Ok(())
}

fn valid_reconnection(reconnection: &super::DelaunayVolumeSliverReconnection) -> bool {
    let expected_counts = match reconnection.kind {
        super::DelaunayVolumeReconnectionKind::TwoToThreeFace => (2, 3),
        super::DelaunayVolumeReconnectionKind::ThreeToTwoEdge => (3, 2),
        super::DelaunayVolumeReconnectionKind::EdgeStar => {
            let removed = reconnection.removed_tetrahedron_node_identities.len();
            (removed, removed.saturating_sub(2).saturating_mul(2))
        }
    };
    reconnection.removed_tetrahedron_node_identities.len() == expected_counts.0
        && reconnection.created_tetrahedron_node_identities.len() == expected_counts.1
        && reconnection
            .removed_tetrahedron_node_identities
            .contains(&reconnection.source_tetrahedron_node_identities)
        && reconnection
            .removed_tetrahedron_node_identities
            .iter()
            .chain(&reconnection.created_tetrahedron_node_identities)
            .all(|tetrahedron| {
                !tetrahedron.contains(&StableDigest::ZERO)
                    && tetrahedron.windows(2).all(|pair| pair[0] < pair[1])
            })
}
