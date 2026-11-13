//! Algebraic numeric-array inventory using the shared codec's actual scalar cost.
//! No array proportional to the requested count is allocated or traversed.
//!
//! The profile sizes the array; it does not cap it. The run's admitted memory and scratch
//! budget is the only ceiling on a planned array (`admit_resource_plan` compares the plan's
//! request, which includes this profile, with the worker inventory). The one remaining bound
//! is the platform's largest single allocation, which no budget can raise.
use super::ResourcePlanningError;
use runmat_canonical_codec::{
    profile_serialized, CanonicalLimits, SerializedCanonicalProfile, PLATFORM_MAXIMUM_ENCODED_BYTES,
};

/// Profiling limits. Collections are bounded by the platform allocation ceiling; the run's
/// admitted budget governs the actual size. String and nesting bounds are unused by numeric
/// arrays.
const PROFILE_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 32);

pub(super) fn numeric_inventory_profile(
    count: u64,
) -> Result<SerializedCanonicalProfile, ResourcePlanningError> {
    let overflow = || ResourcePlanningError::Overflow("numeric serialization inventory");
    let profile = |v: &[f64]| {
        profile_serialized(v, PROFILE_LIMITS)
            .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))
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
    if wire > PLATFORM_MAXIMUM_ENCODED_BYTES as u64 {
        return Err(ResourcePlanningError::CanonicalWireCapacity {
            required: wire,
            maximum: PLATFORM_MAXIMUM_ENCODED_BYTES as u64,
        });
    }
    if wire > runmat_types::wire_integer::MAX_WIRE_INTEGER
        || resident > runmat_types::wire_integer::MAX_WIRE_INTEGER
    {
        return Err(ResourcePlanningError::Invalid(
            "numeric inventory exceeds canonical shape capacity".into(),
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
    //   transitions and rejects arrays beyond admitted canonical capacity without visiting them. Domain
    //   metadata and transport remain separately reserved. This is structural codec accounting, not
    //   measured RSS or whole study admission.
    #[test]
    fn count_only_inventory_covers_real_codec_headers_and_large_arrays() {
        let limits = PROFILE_LIMITS;
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
        // No fixed item or byte constant caps the profile: the drone arm's retained modal
        // basis (290,919,200 values) and a ten-billion-value array both profile; the run's
        // admitted budget decides whether they fit.
        let arm = numeric_inventory_profile(290_919_200).unwrap();
        assert!(arm.encoded_bytes_upper_bound > 1024 * 1024 * 1024);
        let large = numeric_inventory_profile(10_000_000_000).unwrap();
        assert!(large.encoded_bytes_upper_bound > 10_000_000_000);
        // The platform allocation ceiling and checked arithmetic still refuse.
        assert!(numeric_inventory_profile(PLATFORM_MAXIMUM_ENCODED_BYTES as u64 / 8).is_err());
        assert!(numeric_inventory_profile(u64::MAX).is_err());
    }
}
