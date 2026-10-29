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
pub const CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT: u64 = 64 * 1024;

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

/// Resident estimate for directly encoding an owned value whose encoded size is bounded by
/// `encoded_bytes`. The direct encoder reserves the output once and builds no value tree; it
/// holds at most out-of-order child buffers (copies of bytes that also reach the output) and
/// the sorted keys of live maps (also encoded bytes). Output, held buffers and key metadata are
/// each bounded by the encoded size, plus the fixed per-contract allowance.
pub fn projected_canonical_encoding_resident_bytes_estimate(
    encoded_bytes: u64,
    contract_count: u64,
) -> Option<u64> {
    encoded_bytes.checked_mul(3)?.checked_add(
        contract_count.checked_mul(CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT)?,
    )
}

/// Resident estimate for directly decoding canonical payloads of at most `encoded_bytes` into
/// typed values, before the bytes exist. Each scalar occupies at least one encoded byte and at
/// most eight typed bytes; each array or string header occupies at least one encoded byte and
/// one 24-byte collection header; maps decode into inline structs. The validated bytes stay
/// resident alongside, plus the fixed per-contract allowance.
pub fn projected_canonical_typed_decode_resident_bytes_estimate(
    encoded_bytes: u64,
    contract_count: u64,
) -> Option<u64> {
    encoded_bytes.checked_mul(25)?.checked_add(
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

/// Same structural inventory as contiguous profiling, without a joined payload buffer.
pub fn canonical_chunked_payload_materialization_resident_bytes_estimate<'a>(
    chunks: impl IntoIterator<Item = &'a [u8]>,
    control: &dyn crate::CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    let (profile, encoded_bytes) = value::chunked_profile(chunks, control)?;
    materialization_resident_bytes_estimate(profile, encoded_bytes)
}

/// Typed resident estimate for directly decoding the bare canonical value into
/// typed collections: one word per scalar, a vector header per array or string,
/// string bytes, and no retained map keys (maps decode into typed structs).
/// Direct decoding builds no JSON tree; this is the matching requested estimate.
pub fn canonical_value_typed_resident_bytes_estimate(
    value: &[u8],
    control: &dyn crate::CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    let (profile, _) = crate::value::value_profile(value, control)?;
    typed_resident_bytes_estimate(profile)
}

/// Typed direct-decode estimate of a chunked canonical payload (envelope included), with the
/// encoded bytes the decoder validates in place. Decoding builds typed collections directly and
/// never a value tree.
pub fn canonical_chunked_payload_typed_resident_bytes_estimate<'a>(
    chunks: impl IntoIterator<Item = &'a [u8]>,
    control: &dyn crate::CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    let (profile, encoded_bytes) = crate::value::chunked_profile(chunks, control)?;
    typed_resident_bytes_estimate(profile)?
        .checked_add(encoded_bytes)
        .ok_or_else(|| {
            CanonicalCodecError::resource_limit("typed decode estimate", "byte count overflow")
        })
}

/// Typed direct-decode estimate of a contiguous canonical payload: the same authority the
/// scheduler's input facts use for chunked inputs, for producers that profile their own wire.
pub fn canonical_payload_typed_resident_bytes_estimate(
    encoded: &[u8],
) -> Result<u64, CanonicalCodecError> {
    canonical_chunked_payload_typed_resident_bytes_estimate(
        std::iter::once(encoded),
        &crate::NeverCancelled,
    )
}

pub(crate) fn typed_resident_bytes_estimate(
    profile: value::MaterializationProfile,
) -> Result<u64, CanonicalCodecError> {
    let overflow =
        || CanonicalCodecError::resource_limit("typed decode estimate", "byte count overflow");
    let scalars = profile
        .value_nodes
        .checked_sub(profile.heap_containers)
        .and_then(|nodes| nodes.checked_sub(profile.maps))
        .ok_or_else(overflow)?;
    scalars
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(profile.heap_containers.checked_mul(24)?))
        .and_then(|bytes| bytes.checked_add(profile.value_string_bytes))
        .and_then(|bytes| bytes.checked_add(CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT))
        .ok_or_else(overflow)
}

/// Profile the bare canonical value supplied to decode_contract_with_admission.
/// Envelope validation belongs to that decoder; this never prepends or copies it.
/// The same structural scanner and cooperative checkpoints serve both forms.
pub fn canonical_value_materialization_resident_bytes_estimate(
    value: &[u8],
    control: &dyn crate::CanonicalCodecControl,
) -> Result<u64, CanonicalCodecError> {
    let (profile, bytes) = crate::value::value_profile(value, control)?;
    materialization_resident_bytes_estimate(profile, bytes)
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
