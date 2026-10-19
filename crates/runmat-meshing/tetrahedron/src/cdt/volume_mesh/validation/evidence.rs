use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{quality::predicate::Point3Construction, StableDigest};

use super::super::{
    error, DelaunayVolumeMesh, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshOptions, DelaunayVolumeMeshStage,
};
use crate::cdt::{DelaunayConstraints, DelaunayVolumeNode, DelaunayVolumeSliverMutation};

pub(super) fn validate_facet_insertions(
    constraints: &DelaunayConstraints,
    mesh: &DelaunayVolumeMesh,
    options: DelaunayVolumeMeshOptions,
) -> Result<(), DelaunayVolumeMeshError> {
    let known = lineage_identities(mesh);
    let mut per_constraint = BTreeMap::<u32, u64>::new();
    let mut previous = None;
    for insertion in &mesh.facet_recovery_insertions {
        let key = (
            insertion.constraint_index,
            insertion.support_node_identities,
            insertion.insertion_round,
            insertion.candidate_rank,
            insertion.node_identity,
        );
        let count = per_constraint
            .entry(insertion.constraint_index)
            .or_default();
        *count += 1;
        let inserted_node = mesh
            .topology
            .nodes
            .binary_search_by_key(&insertion.node_identity, |node| node.identity)
            .ok()
            .map(|index| mesh.topology.nodes[index]);
        if insertion.constraint_index as usize >= constraints.facets.len()
            || *count > options.carving.facet_recovery.maximum_cavity_steiner_nodes
            || insertion
                .support_node_identities
                .contains(&StableDigest::ZERO)
            || insertion
                .support_node_identities
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || insertion.node_identity == StableDigest::ZERO
            || inserted_node.is_none()
            || previous.is_some_and(|prior| prior >= key)
            || insertion.node_identity
                != crate::cdt::facet_recovery::facet_steiner_identity(
                    insertion.support_node_identities,
                    insertion.insertion_round,
                    insertion.candidate_rank,
                )
            || insertion
                .support_node_identities
                .iter()
                .chain(std::iter::once(&insertion.node_identity))
                .any(|identity| !known.contains(identity))
        {
            return Err(error(
                DelaunayVolumeMeshStage::FacetRecovery,
                DelaunayVolumeMeshErrorKind::InvalidTopology,
                "facet-recovery insertion evidence is noncanonical or does not resolve to mesh lineage",
            ));
        }
        previous = Some(key);
    }
    Ok(())
}

pub(super) fn validate_mutations(
    constraints: &DelaunayConstraints,
    mesh: &DelaunayVolumeMesh,
    options: DelaunayVolumeMeshOptions,
) -> Result<(), DelaunayVolumeMeshError> {
    let insertion_count = mesh.refinement_insertions.len() as u64;
    let treatment_count = mesh.sliver_mutations.len() as u64;
    let maximum_treatments = insertion_count
        .saturating_add(1)
        .saturating_mul(options.sliver.maximum_passes);
    if insertion_count > options.refinement.maximum_insertions
        || treatment_count > maximum_treatments
    {
        return Err(invalid_mutations());
    }

    // Replay identities backwards from the published topology. This verifies that every ordered
    // insertion, relocation, and deletion has an exact predecessor without reconstructing CAD or
    // rerunning tetrahedralization.
    let mut identities = mesh
        .topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    for mutation in mesh.sliver_mutations.iter().rev() {
        match mutation {
            DelaunayVolumeSliverMutation::Relocated(relocation) => {
                validate_source_tetrahedron(
                    relocation.source_node_identity,
                    relocation.source_tetrahedron_node_identities,
                    &identities,
                )?;
                validate_node(relocation.replacement_node)?;
                if !crate::cdt::volume_sliver::relocation_identity_is_valid(relocation)
                    || !identities.remove(&relocation.replacement_node.identity)
                    || !identities.insert(relocation.source_node_identity)
                {
                    return Err(invalid_mutations());
                }
            }
            DelaunayVolumeSliverMutation::Deleted(deletion) => {
                validate_source_tetrahedron(
                    deletion.source_node_identity,
                    deletion.source_tetrahedron_node_identities,
                    &identities,
                )?;
                if !identities.insert(deletion.source_node_identity) {
                    return Err(invalid_mutations());
                }
            }
            DelaunayVolumeSliverMutation::Reconnected(reconnection) => {
                if reconnection
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
                    .any(|identity| !identities.contains(identity))
                {
                    return Err(invalid_mutations());
                }
            }
            DelaunayVolumeSliverMutation::Inserted(insertion) => {
                validate_node(insertion.inserted_node)?;
                if !crate::cdt::volume_sliver::insertion_identity_is_valid(insertion)
                    || !identities.remove(&insertion.inserted_node.identity)
                    || insertion
                        .source_tetrahedron_node_identities
                        .iter()
                        .chain(&insertion.source_edge_node_identities)
                        .any(|identity| !identities.contains(identity))
                {
                    return Err(invalid_mutations());
                }
            }
        }
    }
    for node in mesh.refinement_insertions.iter().rev() {
        validate_node(*node)?;
        if !identities.remove(&node.identity) {
            return Err(invalid_mutations());
        }
    }
    if constraints
        .nodes
        .iter()
        .any(|node| !identities.contains(&node.identity))
    {
        return Err(invalid_mutations());
    }
    Ok(())
}

fn validate_source_tetrahedron(
    source: StableDigest,
    mut tetrahedron: [StableDigest; 4],
    current: &BTreeSet<StableDigest>,
) -> Result<(), DelaunayVolumeMeshError> {
    tetrahedron.sort_unstable();
    if source == StableDigest::ZERO
        || !tetrahedron.contains(&source)
        || tetrahedron.windows(2).any(|pair| pair[0] >= pair[1])
        || current.contains(&source)
        || tetrahedron
            .iter()
            .filter(|identity| **identity != source)
            .any(|identity| !current.contains(identity))
    {
        return Err(invalid_mutations());
    }
    Ok(())
}

fn validate_node(node: DelaunayVolumeNode) -> Result<(), DelaunayVolumeMeshError> {
    if node.identity == StableDigest::ZERO
        || node
            .coordinates_m
            .iter()
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err(invalid_mutations());
    }
    match node.construction {
        Point3Construction::Explicit => {}
        Point3Construction::Segment {
            endpoints,
            parameter_numerator,
            parameter_exponent,
        } => {
            if endpoints
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
                || parameter_exponent >= 63
                || parameter_numerator > (1_u64 << parameter_exponent)
            {
                return Err(invalid_mutations());
            }
        }
        Point3Construction::TriangleDyadic {
            vertices,
            weights_numerator,
            weight_exponent,
        } => {
            if vertices
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
                || weight_exponent >= 63
                || weights_numerator
                    .into_iter()
                    .try_fold(0_u64, u64::checked_add)
                    != Some(1_u64 << weight_exponent)
                || weights_numerator.contains(&0)
            {
                return Err(invalid_mutations());
            }
        }
        Point3Construction::SimplexDyadic {
            vertices,
            weights_numerator,
            vertex_count,
            weight_exponent,
        } => {
            let active = usize::from(vertex_count);
            if active == 0
                || active > vertices.len()
                || vertices[..active]
                    .iter()
                    .flatten()
                    .any(|coordinate| !coordinate.is_finite())
                || weight_exponent >= 63
                || weights_numerator[..active]
                    .iter()
                    .copied()
                    .try_fold(0_u64, u64::checked_add)
                    != Some(1_u64 << weight_exponent)
                || weights_numerator[active..]
                    .iter()
                    .any(|weight| *weight != 0)
            {
                return Err(invalid_mutations());
            }
        }
        Point3Construction::TetrahedronCentroid { vertices } => {
            if vertices
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
            {
                return Err(invalid_mutations());
            }
        }
    }
    Ok(())
}

fn lineage_identities(mesh: &DelaunayVolumeMesh) -> BTreeSet<StableDigest> {
    let mut identities = mesh
        .topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    identities.extend(mesh.refinement_insertions.iter().map(|node| node.identity));
    for mutation in &mesh.sliver_mutations {
        match mutation {
            DelaunayVolumeSliverMutation::Relocated(relocation) => {
                identities.insert(relocation.source_node_identity);
                identities.insert(relocation.replacement_node.identity);
                identities.extend(relocation.source_tetrahedron_node_identities);
            }
            DelaunayVolumeSliverMutation::Deleted(deletion) => {
                identities.insert(deletion.source_node_identity);
                identities.extend(deletion.source_tetrahedron_node_identities);
            }
            DelaunayVolumeSliverMutation::Reconnected(reconnection) => {
                identities.extend(reconnection.source_tetrahedron_node_identities);
                identities.extend(
                    reconnection
                        .removed_tetrahedron_node_identities
                        .iter()
                        .flatten()
                        .copied(),
                );
                identities.extend(
                    reconnection
                        .created_tetrahedron_node_identities
                        .iter()
                        .flatten()
                        .copied(),
                );
            }
            DelaunayVolumeSliverMutation::Inserted(insertion) => {
                identities.insert(insertion.inserted_node.identity);
                identities.extend(insertion.source_tetrahedron_node_identities);
                identities.extend(insertion.source_edge_node_identities);
            }
        }
    }
    identities
}

fn invalid_mutations() -> DelaunayVolumeMeshError {
    error(
        DelaunayVolumeMeshStage::Refinement,
        DelaunayVolumeMeshErrorKind::InvalidTopology,
        "refinement and optimization evidence do not form a valid ordered identity lineage",
    )
}
