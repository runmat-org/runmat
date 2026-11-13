use super::super::*;
use serde::Serialize;

fn actual_body_size<T: Serialize>(value: &T) -> u64 {
    let prefix = b"width-test\0";
    let encoded = crate::encode_contract(
        prefix,
        "widths/v1",
        value,
        CanonicalLimits::new(1 << 20, 1 << 20, 1 << 20, 64),
    )
    .unwrap();
    let mut decoder = minicbor::Decoder::new(&encoded[prefix.len()..]);
    assert_eq!(decoder.array().unwrap(), Some(2));
    assert_eq!(decoder.str().unwrap(), "widths/v1");
    (encoded.len() - prefix.len() - decoder.position()) as u64
}
fn check<T: Serialize>(value: &T) {
    let profile =
        profile_canonical_encoding(value, CanonicalLimits::new(1 << 20, 1 << 20, 1 << 20, 64))
            .unwrap();
    assert_eq!(
        profile.encoded_bytes_upper_bound - ENVELOPE_BYTES,
        actual_body_size(value)
    );
    assert!(profile.resident_workspace_bytes_estimate >= profile.encoded_bytes_upper_bound);
}

#[test]
fn profile_tracks_actual_integer_headers_and_preserves_float_widths() {
    for value in [
        0u64,
        23,
        24,
        255,
        256,
        65535,
        65536,
        u32::MAX as u64,
        u32::MAX as u64 + 1,
        u64::MAX,
    ] {
        check(&value);
    }
    for value in [
        i64::MIN,
        -65537,
        -65536,
        -257,
        -256,
        -25,
        -24,
        -1,
        0,
        i64::MAX,
    ] {
        check(&value);
    }
    check(&(u64::MAX as i128));
    check(&(i64::MIN as i128));
    check(&[0.0, -0.0, 1.0, -1.0, f64::MIN_POSITIVE, f64::MAX]);
    check(&true);
    check(&false);
    check(&());
}

#[test]
fn byte_arrays_and_sized_strings_use_their_real_wire_widths() {
    let bytes = (0..4096).map(|value| value as u8).collect::<Vec<_>>();
    check(&bytes);
    for length in [0, 23, 24, 255, 256, 4096] {
        check(&"x".repeat(length));
    }
    check(&std::collections::BTreeMap::from([
        (23u64, "value"),
        (24, "other"),
    ]));
    let profile =
        profile_canonical_encoding(&bytes, CanonicalLimits::new(1 << 20, 1 << 20, 1 << 20, 64))
            .unwrap();
    assert!(profile.encoded_bytes_upper_bound < 2 * bytes.len() as u64 + ENVELOPE_BYTES + 9);
}

struct UnknownLength(Vec<u8>);
impl Serialize for UnknownLength {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence = serializer.serialize_seq(None)?;
        for value in &self.0 {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
}
#[test]
fn unknown_length_collections_keep_a_safe_header_bound() {
    let value = UnknownLength(vec![0, 23, 24, 255]);
    let profile =
        profile_canonical_encoding(&value, CanonicalLimits::new(4096, 4096, 4096, 64)).unwrap();
    let body = actual_body_size(&value);
    assert!(profile.encoded_bytes_upper_bound - ENVELOPE_BYTES >= body);
    assert!(profile.encoded_bytes_upper_bound - ENVELOPE_BYTES <= body + 8);
}
