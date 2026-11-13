//! Bounded canonical CBOR mechanics shared by typed RunMat domain contracts.
//!
//! Domain owners supply a stable prefix, domain string, limits, and semantic validation. This
//! crate owns only deterministic serde-value projection and hostile-input decoding rules.

/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

mod direct_decoding;
mod exact_vec;
pub use exact_vec::{deserialize_exact_vec, ExactVec};
mod direct_encoding;
mod materialization;
use materialization::materialization_resident_bytes_estimate;
pub use materialization::{
    canonical_chunked_payload_materialization_resident_bytes_estimate,
    canonical_chunked_payload_typed_resident_bytes_estimate,
    canonical_payload_materialization_resident_bytes_estimate,
    canonical_payload_typed_resident_bytes_estimate,
    canonical_value_materialization_resident_bytes_estimate,
    canonical_value_typed_resident_bytes_estimate,
    projected_canonical_encoding_resident_bytes_estimate,
    projected_canonical_materialization_resident_bytes_estimate,
    projected_canonical_typed_decode_resident_bytes_estimate,
    CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
};

mod json;
mod typed_resident;
pub use json::{encode_json_bounded, profile_json_bytes, write_json_bounded};
pub use typed_resident::typed_resident_bytes_estimate;

mod controlled_serde;
mod serialized_profile;
mod value;

pub use serialized_profile::{
    profile_canonical_encoding, profile_canonical_encoding_with_control,
    profile_direct_materialization, profile_serialized, profile_serialized_shape,
    profile_serialized_with_control, profile_typed_resident_bytes,
    serialized_materialization_resident_bytes_estimate, CanonicalEncodingProfile,
    SerializedCanonicalProfile,
};

use serde::{de::DeserializeOwned, Serialize};

pub trait CanonicalCodecControl: Send + Sync {
    /// Retain the originating probe site through domain adapters.
    #[track_caller]
    fn is_cancelled(&self) -> bool;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NeverCancelled;

impl CanonicalCodecControl for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Largest canonical encoding one process can hold. A Rust allocation is bounded by
/// `isize::MAX` bytes: 2 GiB - 1 on wasm32 (whose address space is 4 GiB), and beyond any
/// machine's memory on 64-bit targets. Run-data contracts use this as their codec ceiling; the
/// run's admitted budget gates their actual size where the bytes are produced or materialized.
pub const PLATFORM_MAXIMUM_ENCODED_BYTES: usize = isize::MAX as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalLimits {
    /// Bound for both trusted owned encoding and hostile decoding.
    pub maximum_encoded_bytes: usize,
    /// Shape limits for hostile decoding and preallocation profiling. Trusted
    /// encoding preserves the established serde projection and byte ceiling.
    pub maximum_collection_items: usize,
    pub maximum_string_bytes: usize,
    pub maximum_nesting_depth: usize,
}

impl CanonicalLimits {
    pub const fn new(
        maximum_encoded_bytes: usize,
        maximum_collection_items: usize,
        maximum_string_bytes: usize,
        maximum_nesting_depth: usize,
    ) -> Self {
        Self {
            maximum_encoded_bytes,
            maximum_collection_items,
            maximum_string_bytes,
            maximum_nesting_depth,
        }
    }

    /// Limits for contracts whose collections grow with admitted content (complete fields,
    /// meshes, byte payloads). Every collection item occupies at least one encoded byte, so
    /// the declared encoded-byte ceiling already bounds every collection length; a separate,
    /// smaller item cap would reject admitted content that fits the byte ceiling.
    pub const fn content_bounded(
        maximum_encoded_bytes: usize,
        maximum_string_bytes: usize,
        maximum_nesting_depth: usize,
    ) -> Self {
        Self::new(
            maximum_encoded_bytes,
            maximum_encoded_bytes,
            maximum_string_bytes,
            maximum_nesting_depth,
        )
    }

    /// Content-bounded limits at the platform allocation ceiling, for run data whose size the
    /// run's admitted budget governs. Grammar, canonical form, string and nesting bounds still
    /// refuse malformed input.
    pub const fn platform_content(
        maximum_string_bytes: usize,
        maximum_nesting_depth: usize,
    ) -> Self {
        Self::content_bounded(
            PLATFORM_MAXIMUM_ENCODED_BYTES,
            maximum_string_bytes,
            maximum_nesting_depth,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalCodecErrorKind {
    Invalid,
    ResourceLimit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalCodecError {
    pub kind: CanonicalCodecErrorKind,
    pub field: &'static str,
    pub reason: String,
}

impl CanonicalCodecError {
    pub(crate) fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self {
            kind: CanonicalCodecErrorKind::Invalid,
            field,
            reason: reason.into(),
        }
    }
    pub(crate) fn resource_limit(field: &'static str, reason: impl Into<String>) -> Self {
        Self {
            kind: CanonicalCodecErrorKind::ResourceLimit,
            field,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for CanonicalCodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for CanonicalCodecError {}

pub fn encode_contract<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    contract: &T,
    limits: CanonicalLimits,
) -> Result<Vec<u8>, CanonicalCodecError> {
    encode_contract_with_control(prefix, domain, contract, limits, &NeverCancelled)
}

pub fn encode_contract_with_control<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    contract: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<Vec<u8>, CanonicalCodecError> {
    validate_envelope(prefix, domain)?;
    direct_encoding::encode(prefix, domain, contract, limits, control)
}

/// Hand the exact bytes of `encode_contract` to `sink`, in order, without materializing the
/// encoding. Use it to digest a contract whose encoding is never stored: the digest equals the
/// digest of `encode_contract`'s output. Returns the encoded byte count.
pub fn stream_contract<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    contract: &T,
    limits: CanonicalLimits,
    sink: &mut dyn FnMut(&[u8]),
) -> Result<u64, CanonicalCodecError> {
    stream_contract_with_control(prefix, domain, contract, limits, &NeverCancelled, sink)
}

pub fn stream_contract_with_control<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    contract: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    sink: &mut dyn FnMut(&[u8]),
) -> Result<u64, CanonicalCodecError> {
    validate_envelope(prefix, domain)?;
    direct_encoding::stream(prefix, domain, contract, limits, control, sink)
}

/// Reserve a previously profiled output bound once, avoiding geometric growth
/// slack in retained records. The capacity is not resource or semantic authority;
/// callers admit workspace separately and verify the returned profile bounds.
pub fn encode_contract_with_capacity<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    contract: &T,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    capacity: usize,
) -> Result<Vec<u8>, CanonicalCodecError> {
    validate_envelope(prefix, domain)?;
    direct_encoding::encode_with_capacity(prefix, domain, contract, limits, control, Some(capacity))
}

pub fn decode_contract<T: DeserializeOwned>(
    prefix: &[u8],
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
) -> Result<T, CanonicalCodecError> {
    decode_contract_with_control(prefix, domain, bytes, limits, &NeverCancelled)
}

pub fn decode_contract_with_control<T: DeserializeOwned>(
    prefix: &[u8],
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<T, CanonicalCodecError> {
    decode_contract_with_admission(prefix, domain, bytes, limits, control, |_| Ok(()))
}

/// Validate hostile grammar and limits, then admit domain-owned materialization
/// before the first typed visitor runs. The callback borrows only the validated
/// contract value; decoding reuses that same grammar admission and original bytes.
pub fn decode_contract_with_admission<T: DeserializeOwned>(
    prefix: &[u8],
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    admit: impl FnOnce(&[u8]) -> Result<(), CanonicalCodecError>,
) -> Result<T, CanonicalCodecError> {
    let value = validated_contract_value(prefix, domain, bytes, limits, control)?;
    admit(value)?;
    direct_decoding::decode_value(value, control)
}

/// Validate canonical grammar, framing, and limits without allocating a value tree.
/// The borrowed value is suitable for a domain-owned typed storage inventory;
/// it grants no semantic authority and remains tied to the original immutable bytes.
pub fn validated_contract_value<'a>(
    prefix: &[u8],
    domain: &str,
    bytes: &'a [u8],
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<&'a [u8], CanonicalCodecError> {
    validate_envelope(prefix, domain)?;
    checkpoint(control)?;
    if bytes.len() > limits.maximum_encoded_bytes {
        return Err(invalid(
            "canonical decoding",
            format!(
                "contract `{domain}` contains {} bytes, exceeding its {}-byte limit",
                bytes.len(),
                limits.maximum_encoded_bytes
            ),
        ));
    }
    let payload = bytes
        .strip_prefix(prefix)
        .ok_or_else(|| invalid("canonical decoding", "codec domain prefix is missing"))?;
    direct_decoding::validated_value(domain, payload, limits, control)
}

/// Exposed for hostile-codec conformance tests that must deliberately construct noncanonical
/// envelopes. Production domain code should use `encode_contract`.
#[doc(hidden)]
pub fn encode_json_value(
    encoder: &mut minicbor::Encoder<&mut Vec<u8>>,
    value: &serde_json::Value,
) -> Result<(), CanonicalCodecError> {
    value::encode_value(encoder, value, &NeverCancelled)
}

#[cfg(test)]
fn encode_value_envelope(
    prefix: &[u8],
    domain: &str,
    value: &serde_json::Value,
    control: &dyn CanonicalCodecControl,
) -> Result<Vec<u8>, CanonicalCodecError> {
    let mut encoded = prefix.to_vec();
    let mut encoder = minicbor::Encoder::new(&mut encoded);
    encoder
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    value::encode_value(&mut encoder, value, control)?;
    Ok(encoded)
}

#[track_caller]
pub(crate) fn checkpoint(control: &dyn CanonicalCodecControl) -> Result<(), CanonicalCodecError> {
    if control.is_cancelled() {
        return Err(CanonicalCodecError::invalid(
            "canonical codec",
            "operation was cancelled",
        ));
    }
    Ok(())
}

fn validate_envelope(prefix: &[u8], domain: &str) -> Result<(), CanonicalCodecError> {
    if prefix.is_empty() || prefix.len() > 128 || !prefix.ends_with(&[0]) {
        return Err(invalid(
            "canonical prefix",
            "must be 1..=128 bytes and end with a NUL domain separator",
        ));
    }
    if domain.is_empty()
        || domain.len() > 128
        || !domain.is_ascii()
        || domain.chars().any(char::is_whitespace)
    {
        return Err(invalid(
            "canonical domain",
            "must be 1..=128 non-whitespace ASCII bytes",
        ));
    }
    Ok(())
}

#[cfg(test)]
fn require_array_length(
    length: Result<Option<u64>, minicbor::decode::Error>,
    expected: u64,
    field: &'static str,
) -> Result<(), CanonicalCodecError> {
    if length.map_err(decoding_error)? != Some(expected) {
        return Err(invalid(
            field,
            format!("expected a definite array of length {expected}"),
        ));
    }
    Ok(())
}

pub(crate) fn encoding_error<E: std::fmt::Display>(error: E) -> CanonicalCodecError {
    invalid("canonical encoding", error)
}

#[cfg(test)]
pub(crate) fn decoding_error<E: std::fmt::Display>(error: E) -> CanonicalCodecError {
    invalid("canonical decoding", error)
}

fn invalid(field: &'static str, reason: impl ToString) -> CanonicalCodecError {
    CanonicalCodecError::invalid(field, reason.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use minicbor::Encoder;
    use serde::{Deserialize, Serialize};

    use super::*;

    const PREFIX: &[u8] = b"runmat-codec-test/v1\0";
    const DOMAIN: &str = "test.contract/v1";
    const LIMITS: CanonicalLimits = CanonicalLimits::new(4096, 32, 128, 8);

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Fixture {
        values: BTreeMap<String, f64>,
    }

    fn fixture() -> Fixture {
        Fixture {
            values: BTreeMap::from([("a".into(), -0.0), ("b".into(), 2.0)]),
        }
    }

    #[test]
    fn canonical_round_trip_preserves_exact_floats() {
        let bytes = encode_contract(PREFIX, DOMAIN, &fixture(), LIMITS).unwrap();
        let decoded: Fixture = decode_contract(PREFIX, DOMAIN, &bytes, LIMITS).unwrap();
        assert_eq!(decoded, fixture());
        assert!(decoded.values["a"].is_sign_negative());
        assert_eq!(
            encode_contract(PREFIX, DOMAIN, &decoded, LIMITS).unwrap(),
            bytes
        );
    }

    #[test]
    fn controlled_codec_observes_cancellation() {
        struct Cancelled;

        impl CanonicalCodecControl for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }

        assert_eq!(
            encode_contract_with_control(PREFIX, DOMAIN, &fixture(), LIMITS, &Cancelled)
                .unwrap_err()
                .field,
            "canonical codec"
        );
        let bytes = encode_contract(PREFIX, DOMAIN, &fixture(), LIMITS).unwrap();
        assert_eq!(
            decode_contract_with_control::<Fixture>(PREFIX, DOMAIN, &bytes, LIMITS, &Cancelled,)
                .unwrap_err()
                .field,
            "canonical codec"
        );
    }

    #[test]
    fn envelope_domain_prefix_and_byte_bounds_fail_closed() {
        let bytes = encode_contract(PREFIX, DOMAIN, &fixture(), LIMITS).unwrap();
        assert_eq!(
            decode_contract::<Fixture>(PREFIX, "test.other/v1", &bytes, LIMITS)
                .unwrap_err()
                .field,
            "canonical decoding domain"
        );
        assert!(decode_contract::<Fixture>(b"wrong\0", DOMAIN, &bytes, LIMITS).is_err());
        assert!(decode_contract::<Fixture>(
            PREFIX,
            DOMAIN,
            &bytes,
            CanonicalLimits::new(bytes.len() - 1, 32, 128, 8),
        )
        .is_err());
    }

    #[test]
    fn byte_bound_failure_names_domain_actual_size_and_unchanged_limit() {
        let bytes = encode_contract(PREFIX, DOMAIN, &fixture(), LIMITS).unwrap();
        let limit = bytes.len() - 1;
        let limits = CanonicalLimits::new(limit, 32, 128, 8);
        for error in [
            encode_contract(PREFIX, DOMAIN, &fixture(), limits).unwrap_err(),
            decode_contract::<Fixture>(PREFIX, DOMAIN, &bytes, limits).unwrap_err(),
        ] {
            let message = error.to_string();
            assert!(message.contains(DOMAIN));
            assert!(message.contains(&format!("{} bytes", bytes.len())));
            assert!(message.contains(&format!("{limit}-byte limit")));
        }
    }

    #[test]
    fn decoder_rejects_noncanonical_key_order_and_collection_length() {
        let mut bytes = PREFIX.to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(2)
            .and_then(|encoder| encoder.str(DOMAIN))
            .and_then(|encoder| encoder.map(2))
            .unwrap();
        for key in ["z", "a"] {
            encoder.str(key).unwrap();
            encoder.null().unwrap();
        }
        assert!(decode_contract::<serde_json::Value>(PREFIX, DOMAIN, &bytes, LIMITS).is_err());

        let mut oversized = PREFIX.to_vec();
        let mut encoder = Encoder::new(&mut oversized);
        encoder
            .array(2)
            .and_then(|encoder| encoder.str(DOMAIN))
            .and_then(|encoder| encoder.array(33))
            .unwrap();
        assert!(decode_contract::<Fixture>(PREFIX, DOMAIN, &oversized, LIMITS).is_err());
    }

    #[test]
    fn materialization_envelope_is_checked_and_scales_with_contract_count() {
        assert_eq!(
            projected_canonical_materialization_resident_bytes_estimate(1, 1),
            Some(128 + 64 * 1024)
        );
        assert_eq!(
            projected_canonical_materialization_resident_bytes_estimate(2, 3),
            Some(256 + 3 * 64 * 1024)
        );
        assert_eq!(
            projected_canonical_materialization_resident_bytes_estimate(u64::MAX, 1),
            None
        );
        assert_eq!(
            projected_canonical_materialization_resident_bytes_estimate(1, u64::MAX),
            None
        );

        let bytes = encode_contract(PREFIX, DOMAIN, &fixture(), LIMITS).unwrap();
        let structural = canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap();
        assert!(structural > bytes.len() as u64);

        let mut malformed = bytes;
        malformed.push(0);
        assert!(canonical_payload_materialization_resident_bytes_estimate(&malformed).is_err());
    }

    #[test]
    fn numeric_payload_profile_avoids_the_size_only_scaling_cliff() {
        let values = vec![1.0_f64; 100_000];
        let limits = CanonicalLimits::new(2 * 1024 * 1024, 200_000, 128, 8);
        let bytes = encode_contract(PREFIX, "test.numeric/v1", &values, limits).unwrap();
        let profiled = canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap();
        let projected =
            projected_canonical_materialization_resident_bytes_estimate(bytes.len() as u64, 1)
                .unwrap();
        assert!(profiled < projected);
        assert!(profiled > bytes.len() as u64);
    }

    #[test]
    fn projected_envelope_covers_compact_scalar_collections() {
        let limits = CanonicalLimits::new(2 * 1024 * 1024, 400_000, 128, 8);
        let payloads = [
            encode_contract(PREFIX, "test.nulls/v1", &vec![(); 100_000], limits).unwrap(),
            encode_contract(PREFIX, "test.bools/v1", &vec![true; 100_000], limits).unwrap(),
            encode_contract(PREFIX, "test.ints/v1", &vec![0_u8; 100_000], limits).unwrap(),
        ];
        for bytes in payloads {
            let profiled =
                canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap();
            let projected =
                projected_canonical_materialization_resident_bytes_estimate(bytes.len() as u64, 1)
                    .unwrap();
            assert!(profiled <= projected);
        }
    }
}

mod json_materialization;
pub use json_materialization::json_materialization_resident_bytes_estimate;
