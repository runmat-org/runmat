//! Digests and codec limits for contracts whose size grows with the admitted model.
//!
//! A contract encoded only to identify it is hashed from the streamed canonical encoding, so no
//! buffer of its size exists and only the wire-integer range bounds it. A stored contract's size
//! is admitted by its resource plan: either the codec takes the plan's bound
//! (`planned_content_limits`), or the object inventory checks the stored payload against the
//! plan and the codec bounds it only by the wire range (`wire_bounded`). Decoding never
//! preallocates from declared lengths, so admitted input bytes bound every decoded collection.
use runmat_canonical_codec::{stream_contract, CanonicalCodecError, CanonicalLimits};
use runmat_execution::{Digest, DigestStream};
use runmat_types::wire_integer::MAX_WIRE_INTEGER;
use serde::Serialize;

/// Largest encoded size or item count any canonical contract can declare on the wire.
pub const WIRE_BOUND: usize = if (MAX_WIRE_INTEGER as u128) < usize::MAX as u128 {
    MAX_WIRE_INTEGER as usize
} else {
    usize::MAX
};

/// `limits` with the byte and item ceilings raised to the wire bound. String and nesting
/// bounds belong to the contract's format and are kept.
pub const fn wire_bounded(limits: CanonicalLimits) -> CanonicalLimits {
    CanonicalLimits::new(
        WIRE_BOUND,
        WIRE_BOUND,
        limits.maximum_string_bytes,
        limits.maximum_nesting_depth,
    )
}

/// SHA-256 of `encode_contract(prefix, domain, value, ..)`, computed from the streamed
/// encoding. Equal to hashing the materialized bytes; the encoding is never retained.
pub fn canonical_contract_digest<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
) -> Result<Digest, CanonicalCodecError> {
    let mut hash = DigestStream::new();
    stream_contract(prefix, domain, value, wire_bounded(limits), &mut |chunk| {
        hash.update(chunk)
    })?;
    Ok(hash.finish())
}

/// Codec limits for a contract the resource plan admitted at `encoded_bytes`: every collection
/// item occupies at least one encoded byte, so the byte bound also bounds every collection.
pub fn planned_content_limits(
    encoded_bytes: u64,
    format: CanonicalLimits,
) -> Result<CanonicalLimits, CanonicalCodecError> {
    if encoded_bytes > MAX_WIRE_INTEGER {
        return Err(CanonicalCodecError {
            kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
            field: "canonical limits",
            reason: format!("planned contract of {encoded_bytes} bytes exceeds the wire bound"),
        });
    }
    let bytes = usize::try_from(encoded_bytes).map_err(|_| CanonicalCodecError {
        kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
        field: "canonical limits",
        reason: format!("planned contract of {encoded_bytes} bytes exceeds this platform"),
    })?;
    Ok(CanonicalLimits::content_bounded(
        bytes,
        format.maximum_string_bytes,
        format.maximum_nesting_depth,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_canonical_codec::encode_contract;

    #[test]
    fn streamed_digest_equals_the_digest_of_the_materialized_encoding() {
        let value: Vec<(u64, [f64; 3], String)> = (0..100_000)
            .map(|i| (i, [i as f64; 3], format!("entity-{i}")))
            .collect();
        let limits = CanonicalLimits::new(64 << 20, 1 << 20, 64, 8);
        let bytes = encode_contract(b"runmat-test/v1\0", "digest", &value, limits).unwrap();
        assert!(bytes.len() > 1 << 20, "spans several streamed chunks");
        assert_eq!(
            canonical_contract_digest(b"runmat-test/v1\0", "digest", &value, limits).unwrap(),
            Digest::sha256(&bytes)
        );
    }

    /// A contract of `count` 64 KiB byte blocks, produced lazily: nothing of it is held.
    struct Blocks {
        count: usize,
    }
    #[derive(Clone)]
    struct Block;
    impl Serialize for Block {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            s.serialize_bytes(&[0x05; 64 * 1024])
        }
    }
    impl Serialize for Blocks {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            s.collect_seq(std::iter::repeat_n(Block, self.count))
        }
    }

    // witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: a streamed contract above 1 GiB is admitted under a plan bound above its size
    //   and digested without a buffer of its size, and is rejected under a plan bound below its
    //   size; codec accounting only, not a measured study.
    #[test]
    fn contracts_above_one_gibibyte_follow_the_plan_bound() {
        const GIB: u64 = 1 << 30;
        // Each block encodes as 64 KiB one-byte items plus a header: just above 1 GiB in all.
        let contract = Blocks {
            count: 16 * 1024 + 64,
        };
        let format = CanonicalLimits::new(0, 0, 64, 4);
        let mut wire = DigestStream::new();
        let size = stream_contract(
            b"runmat-test/v1\0",
            "large",
            &contract,
            wire_bounded(format),
            &mut |chunk| wire.update(chunk),
        )
        .unwrap();
        assert!(
            size > GIB,
            "the contract exceeds the former fixed 1 GiB ceiling"
        );
        // A plan that admits the contract's size accepts it, digest unchanged.
        let mut planned = DigestStream::new();
        stream_contract(
            b"runmat-test/v1\0",
            "large",
            &contract,
            planned_content_limits(size, format).unwrap(),
            &mut |chunk| planned.update(chunk),
        )
        .unwrap();
        assert_eq!(planned.finish(), wire.finish());
        // A plan one byte short rejects it, without a buffer of its size.
        let rejected = stream_contract(
            b"runmat-test/v1\0",
            "large",
            &contract,
            planned_content_limits(size - 1, format).unwrap(),
            &mut |_| {},
        )
        .unwrap_err();
        assert_eq!(
            rejected.kind,
            runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit
        );
    }

    #[test]
    fn planned_limits_follow_the_plan_up_to_the_wire_bound() {
        let format = CanonicalLimits::new(1, 1, 4096, 32);
        let limits = planned_content_limits(3 << 30, format).unwrap();
        assert_eq!(limits.maximum_encoded_bytes, 3 << 30);
        assert_eq!(limits.maximum_collection_items, 3 << 30);
        assert_eq!(limits.maximum_string_bytes, 4096);
        assert!(planned_content_limits(MAX_WIRE_INTEGER + 1, format).is_err());
    }
}
