//! Bounded canonical CBOR mechanics shared by typed RunMat domain contracts.
//!
//! Domain owners supply a stable prefix, domain string, limits, and semantic validation. This
//! crate owns only deterministic serde-value projection and hostile-input decoding rules.

mod value;

use serde::{de::DeserializeOwned, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalLimits {
    pub maximum_encoded_bytes: usize,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalCodecError {
    pub field: &'static str,
    pub reason: String,
}

impl CanonicalCodecError {
    pub(crate) fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self {
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
    validate_envelope(prefix, domain)?;
    let value =
        serde_json::to_value(contract).map_err(|error| invalid("canonical encoding", error))?;
    let mut encoded = Vec::with_capacity(1024);
    encoded.extend_from_slice(prefix);
    let mut encoder = minicbor::Encoder::new(&mut encoded);
    encoder
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    value::encode_value(&mut encoder, &value)?;
    if encoded.len() > limits.maximum_encoded_bytes {
        return Err(invalid(
            "canonical encoding",
            "encoded contract exceeds its byte limit",
        ));
    }
    Ok(encoded)
}

pub fn decode_contract<T: DeserializeOwned>(
    prefix: &[u8],
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
) -> Result<T, CanonicalCodecError> {
    validate_envelope(prefix, domain)?;
    if bytes.len() > limits.maximum_encoded_bytes {
        return Err(invalid(
            "canonical decoding",
            "encoded contract exceeds its byte limit",
        ));
    }
    let payload = bytes
        .strip_prefix(prefix)
        .ok_or_else(|| invalid("canonical decoding", "codec domain prefix is missing"))?;
    let mut decoder = minicbor::Decoder::new(payload);
    require_array_length(decoder.array(), 2, "canonical envelope")?;
    let actual_domain = decoder.str().map_err(decoding_error)?;
    if actual_domain != domain {
        return Err(invalid(
            "canonical decoding domain",
            format!("expected {domain}, received {actual_domain}"),
        ));
    }
    let value = value::decode_value(&mut decoder, limits, 0)?;
    if decoder.position() != payload.len() {
        return Err(invalid("canonical decoding", "trailing data is forbidden"));
    }
    let canonical = encode_value_envelope(prefix, domain, &value)?;
    if canonical.as_slice() != bytes {
        return Err(invalid(
            "canonical decoding",
            "input is not in canonical form",
        ));
    }
    serde_json::from_value(value).map_err(|error| invalid("canonical decoding", error))
}

/// Exposed for hostile-codec conformance tests that must deliberately construct noncanonical
/// envelopes. Production domain code should use `encode_contract`.
#[doc(hidden)]
pub fn encode_json_value(
    encoder: &mut minicbor::Encoder<&mut Vec<u8>>,
    value: &serde_json::Value,
) -> Result<(), CanonicalCodecError> {
    value::encode_value(encoder, value)
}

fn encode_value_envelope(
    prefix: &[u8],
    domain: &str,
    value: &serde_json::Value,
) -> Result<Vec<u8>, CanonicalCodecError> {
    let mut encoded = prefix.to_vec();
    let mut encoder = minicbor::Encoder::new(&mut encoded);
    encoder
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    value::encode_value(&mut encoder, value)?;
    Ok(encoded)
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
}
