//! Algebraic numeric-array inventory using the shared codec's actual scalar cost.
//! No array proportional to the requested count is allocated or traversed.
use super::ResourcePlanningError;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits, SerializedCanonicalProfile};

pub(super) fn numeric_inventory_profile(
    count: u64,
) -> Result<SerializedCanonicalProfile, ResourcePlanningError> {
    let overflow = || ResourcePlanningError::Overflow("numeric serialization inventory");
    let limits = CanonicalLimits::new(1024 * 1024 * 1024, 100_000_000, 65_536, 32);
    if count > limits.maximum_collection_items as u64 {
        return Err(ResourcePlanningError::Invalid(
            "numeric inventory exceeds canonical shape capacity".into(),
        ));
    }
    let profile = |v: &[f64]| {
        profile_serialized(v, limits).map_err(|e| ResourcePlanningError::Invalid(e.to_string()))
    };
    let empty = profile(&[])?;
    let one = profile(&[f64::MAX])?;
    // A sequence header can grow from one to nine bytes. Two wire buffers are
    // included by the materialization owner; all other scalar increments are exact.
    let wire_increment = one
        .encoded_bytes_upper_bound
        .checked_sub(empty.encoded_bytes_upper_bound)
        .ok_or_else(overflow)?;
    let resident_increment = one
        .materialization_resident_bytes_estimate
        .checked_sub(empty.materialization_resident_bytes_estimate)
        .ok_or_else(overflow)?;
    let wire = count
        .checked_mul(wire_increment)
        .and_then(|n| n.checked_add(empty.encoded_bytes_upper_bound))
        .and_then(|n| n.checked_add(8))
        .ok_or_else(overflow)?;
    let resident = count
        .checked_mul(resident_increment)
        .and_then(|n| n.checked_add(empty.materialization_resident_bytes_estimate))
        .and_then(|n| n.checked_add(16))
        .ok_or_else(overflow)?;
    if wire > limits.maximum_encoded_bytes as u64 {
        return Err(ResourcePlanningError::Invalid(
            "numeric inventory exceeds canonical wire capacity".into(),
        ));
    }
    Ok(SerializedCanonicalProfile {
        encoded_bytes_upper_bound: wire,
        materialization_resident_bytes_estimate: resident,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Count-only numerical profiling covers real codec arrays across header-width
    //   transitions and plans100million values without constructing or visiting that array. Domain
    //   metadata and transport remain separately reserved. This is structural codec accounting, not
    //   measured RSS or whole study admission.
    #[test]
    fn count_only_inventory_covers_real_codec_headers_and_large_arrays() {
        let limits = CanonicalLimits::new(1024 * 1024 * 1024, 100_000_000, 65_536, 32);
        for n in [0, 1, 23, 24, 255, 256, 65535, 65536] {
            let values = vec![f64::MAX; n];
            let actual = profile_serialized(&values, limits).unwrap();
            let bound = numeric_inventory_profile(n as u64).unwrap();
            assert!(bound.encoded_bytes_upper_bound >= actual.encoded_bytes_upper_bound);
            assert!(
                bound.materialization_resident_bytes_estimate
                    >= actual.materialization_resident_bytes_estimate
            );
        }
        // This call has constant work/storage, even at a hundred million values.
        assert!(numeric_inventory_profile(100_000_000).is_ok());
        assert!(numeric_inventory_profile(u64::MAX).is_err());
    }
}
