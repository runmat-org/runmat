//! One original point-identity payload shared by encoding and workspace profiling.
use super::*;

fn identity_inputs<'a>(
    topology: Digest,
    ordinal: u64,
    position: [f64; 3],
    element: StableDigest,
    node_indices: &'a [usize],
    weights: &'a [f64],
) -> impl Serialize + 'a {
    (topology, ordinal, position, element, node_indices, weights)
}

pub(in crate::results::observation) fn derive_resolved_point_identity(
    topology_digest: Digest,
    sample: &ResolvedObservationPoint,
) -> Result<StableDigest, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_PREFIX,
        "resolved-observation-point",
        &identity_inputs(
            topology_digest,
            sample.ordinal,
            sample.position_m,
            sample.source_element_identity,
            &sample.source_node_indices,
            &sample.shape_values,
        ),
        OBSERVATION_LIMITS,
    )?;
    Ok(StableDigest::from_bytes(*Digest::sha256(bytes).bytes()))
}

pub(in crate::results::observation) fn resolved_point_identity_resident_bytes(
    node_count: usize,
) -> Result<u64, ObservationError> {
    if node_count > 10 {
        return Err(ObservationError::Limit("solid point node count"));
    }
    let nodes = [usize::MAX; 10];
    let weights = [f64::MAX; 10];
    let profile = runmat_canonical_codec::profile_serialized(
        &identity_inputs(
            Digest::sha256("point profile"),
            u64::MAX,
            [f64::MAX; 3],
            StableDigest::ZERO,
            &nodes[..node_count],
            &weights[..node_count],
        ),
        OBSERVATION_LIMITS,
    )?;
    // Canonical materialization coexists with its encoded identity buffer.
    profile
        .materialization_resident_bytes_estimate
        .checked_add(profile.encoded_bytes_upper_bound)
        .ok_or(ObservationError::Limit("point identity workspace"))
}

#[cfg(test)]
mod tests;
