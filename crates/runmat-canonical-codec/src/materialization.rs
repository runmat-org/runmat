//! Shared resource estimates for canonical projection and hostile payload verification.
use crate::{value, CanonicalCodecError};

/// Deterministic resident-memory estimate for canonical serde-value materialization when only an
/// encoded-size ceiling is known. Domain payloads use the structural profiler below once bytes
/// exist; pre-execution output projection uses this deliberately conservative multiplier.
// Every canonical value node consumes at least one encoded byte. This factor covers two complete
// `serde_json::Value`-sized representations per possible node, worst-density map/string metadata,
// typed materialization, and simultaneous input/output buffers. The fixed allowance covers the
// collection allocators and envelope metadata that do not scale linearly with payload bytes.
const PROJECTED_MATERIALIZATION_BYTES_PER_ENCODED_BYTE: u64 = 128;
pub(crate) const CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT: u64 = 64 * 1024;

pub fn projected_canonical_materialization_resident_bytes_estimate(
    encoded_bytes: u64,
    contract_count: u64,
) -> Option<u64> {
    encoded_bytes
        .checked_mul(PROJECTED_MATERIALIZATION_BYTES_PER_ENCODED_BYTE)?
        .checked_add(
            contract_count.checked_mul(CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT)?,
        )
}

/// Profiles an existing canonical payload without allocating its value tree, then estimates the
/// peak of canonical verification plus typed materialization. The estimate counts actual value
/// nodes, map entries, and string bytes instead of multiplying numeric-array payloads by a generic
/// worst case. Domain execution still records authoritative allocator/scheduler usage.
pub fn canonical_payload_materialization_resident_bytes_estimate(
    encoded: &[u8],
) -> Result<u64, CanonicalCodecError> {
    let profile = value::materialization_profile(encoded)?;
    let encoded_bytes = u64::try_from(encoded.len()).map_err(|_| {
        CanonicalCodecError::resource_limit("materialization estimate", "payload bytes overflow")
    })?;
    materialization_resident_bytes_estimate(profile, encoded_bytes)
}

pub(crate) fn materialization_resident_bytes_estimate(
    profile: value::MaterializationProfile,
    encoded_bytes: u64,
) -> Result<u64, CanonicalCodecError> {
    let value_bytes = profile
        .value_nodes
        .checked_mul(std::mem::size_of::<serde_json::Value>() as u64)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| {
            CanonicalCodecError::resource_limit("materialization estimate", "value bytes overflow")
        })?;
    let map_entry_bytes = profile
        .map_entries
        .checked_mul(
            (std::mem::size_of::<String>()
                + std::mem::size_of::<serde_json::Value>()
                + 4 * std::mem::size_of::<usize>()) as u64,
        )
        .ok_or_else(|| {
            CanonicalCodecError::resource_limit("materialization estimate", "map bytes overflow")
        })?;
    let string_bytes = profile.string_bytes.checked_mul(2).ok_or_else(|| {
        CanonicalCodecError::resource_limit("materialization estimate", "string bytes overflow")
    })?;
    let encoded_bytes = encoded_bytes.checked_mul(2).ok_or_else(|| {
        CanonicalCodecError::resource_limit("materialization estimate", "payload bytes overflow")
    })?;
    value_bytes
        .checked_add(map_entry_bytes)
        .and_then(|bytes| bytes.checked_add(string_bytes))
        .and_then(|bytes| bytes.checked_add(encoded_bytes))
        .and_then(|bytes| bytes.checked_add(CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT))
        .ok_or_else(|| {
            CanonicalCodecError::resource_limit(
                "materialization estimate",
                "resident bytes overflow",
            )
        })
}
