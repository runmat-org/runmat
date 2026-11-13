use super::*;
mod admission;
mod visitors;
use crate::{decode_contract, encode_contract, NeverCancelled};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};
const PREFIX: &[u8] = b"direct-read\0";
const DOMAIN: &str = "direct-read/v1";
const LIMITS: CanonicalLimits = CanonicalLimits::new(16 << 20, 1_000_000, 65_536, 64);

fn original<T: DeserializeOwned>(
    bytes: &[u8],
    limits: CanonicalLimits,
) -> Result<T, CanonicalCodecError> {
    let payload = bytes
        .strip_prefix(PREFIX)
        .ok_or_else(|| invalid("prefix"))?;
    if bytes.len() > limits.maximum_encoded_bytes {
        return Err(invalid("bytes"));
    }
    let mut decoder = minicbor::Decoder::new(payload);
    crate::require_array_length(decoder.array(), 2, "envelope")?;
    if decoder.str().map_err(invalid)? != DOMAIN {
        return Err(invalid("domain"));
    }
    let value = crate::value::decode_value(&mut decoder, limits, 0, &NeverCancelled)?;
    if decoder.position() != payload.len() {
        return Err(invalid("trailing"));
    }
    if crate::encode_value_envelope(PREFIX, DOMAIN, &value, &NeverCancelled)? != bytes {
        return Err(invalid("canonical"));
    }
    crate::controlled_serde::from_value(value, &NeverCancelled).map_err(invalid)
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Shape {
    Unit,
    Scalar(f64),
    Tuple(bool, u64),
    Fields { a: Option<String>, values: Vec<u8> },
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "content")]
enum Tagged {
    List(Vec<Shape>),
    None,
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum Alternative {
    Count(u64),
    Label(String),
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Fixture {
    shape: Tagged,
    alternative: Alternative,
    keys: BTreeMap<i128, Vec<f64>>,
    booleans: BTreeMap<bool, u64>,
    tuple: (char, i128, u128, Option<String>),
}
#[test]
fn typed_decode_preserves_original_json_projection_and_wire() {
    let fixture = Fixture {
        shape: Tagged::List(vec![
            Shape::Unit,
            Shape::Scalar(-0.0),
            Shape::Tuple(true, u64::MAX),
            Shape::Fields {
                a: Some("µλ".into()),
                values: vec![0, 127, 255],
            },
        ]),
        alternative: Alternative::Label("mesh".into()),
        keys: BTreeMap::from([(i128::MIN, vec![-0.0, 1e-200]), (i128::MAX, vec![1e200])]),
        booleans: BTreeMap::from([(false, 0), (true, u64::MAX)]),
        tuple: ('λ', i64::MIN.into(), u64::MAX.into(), None),
    };
    let bytes = encode_contract(PREFIX, DOMAIN, &fixture, LIMITS).unwrap();
    let actual: Fixture = decode_contract(PREFIX, DOMAIN, &bytes, LIMITS).unwrap();
    assert_eq!(actual, original::<Fixture>(&bytes, LIMITS).unwrap());
    assert_eq!(actual, fixture);
    assert!(actual.keys[&i128::MIN][0].is_sign_negative());
    assert_eq!(
        encode_contract(PREFIX, DOMAIN, &actual, LIMITS).unwrap(),
        bytes
    );
}
// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: All truncations and single-bit edits of a mixed canonical fixture compare admission
//   and decoded values with the original complete decode/re-encode oracle. The family covers
//   representative hostile grammar; it does not establish exhaustive hostile-input coverage or
//   measured native memory.
#[test]
fn hostile_byte_edits_and_truncation_match_original_acceptance() {
    let source = serde_json::json!({"a":[null,true,false,-1,0,u64::MAX,-0.0,1e200,"λ"],"b":{"first":[],"last":{}}});
    let bytes = encode_contract(PREFIX, DOMAIN, &source, LIMITS).unwrap();
    let compare = |bytes: &[u8]| {
        let actual = decode_contract::<serde_json::Value>(PREFIX, DOMAIN, bytes, LIMITS);
        let reference = original::<serde_json::Value>(bytes, LIMITS);
        assert_eq!(
            actual.is_ok(),
            reference.is_ok(),
            "different acceptance for {bytes:?}"
        );
        if let (Ok(actual), Ok(reference)) = (actual, reference) {
            assert_eq!(actual, reference);
        }
    };
    for end in 0..=bytes.len() {
        compare(&bytes[..end]);
    }
    for position in 0..bytes.len() {
        for bit in 0..8 {
            let mut edited = bytes.clone();
            edited[position] ^= 1 << bit;
            compare(&edited);
        }
    }
    for extra in [0, 0xff] {
        let mut edited = bytes.clone();
        edited.push(extra);
        compare(&edited);
    }
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Byte, collection, string and depth rejection plus malformed terminal UTF-8 occur
//   before an observable typed constructor is entered. This verifies admission ordering, not a
//   universal typed allocation bound or a lower whole-task resource reservation.
#[test]
fn shape_limits_apply_before_any_typed_construction() {
    static CONSTRUCTED: AtomicUsize = AtomicUsize::new(0);
    struct Count;
    impl<'de> Deserialize<'de> for Count {
        fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
            CONSTRUCTED.fetch_add(1, Ordering::SeqCst);
            let _ = serde_json::Value::deserialize(decoder)?;
            Ok(Self)
        }
    }
    let value = vec![vec!["long-string"; 4]; 3];
    let bytes = encode_contract(PREFIX, DOMAIN, &value, LIMITS).unwrap();
    for limits in [
        CanonicalLimits::new(bytes.len() - 1, 10, 100, 64),
        CanonicalLimits::new(bytes.len(), 3, 100, 64),
        CanonicalLimits::new(bytes.len(), 10, 3, 64),
        CanonicalLimits::new(bytes.len(), 10, 100, 1),
    ] {
        assert!(decode_contract::<Count>(PREFIX, DOMAIN, &bytes, limits).is_err());
        assert_eq!(CONSTRUCTED.load(Ordering::SeqCst), 0);
    }
    let mut corrupt = bytes.clone();
    *corrupt.last_mut().unwrap() = 0xff;
    assert!(decode_contract::<Count>(PREFIX, DOMAIN, &corrupt, LIMITS).is_err());
    assert_eq!(CONSTRUCTED.load(Ordering::SeqCst), 0);
}
#[test]
fn cancellation_interrupts_validation_and_typed_numeric_reads() {
    struct Cancel(AtomicUsize, usize);
    impl CanonicalCodecControl for Cancel {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::SeqCst) >= self.1
        }
    }
    let value = vec![1.25_f64; 65536];
    let bytes = encode_contract(PREFIX, DOMAIN, &value, LIMITS).unwrap();
    // The latter threshold reaches typed deserialization after complete preflight.
    for threshold in [20, 65560] {
        let error = crate::decode_contract_with_control::<Vec<f64>>(
            PREFIX,
            DOMAIN,
            &bytes,
            LIMITS,
            &Cancel(AtomicUsize::new(0), threshold),
        )
        .unwrap_err();
        assert_eq!(error.field, "canonical codec");
        assert_eq!(error.reason, "operation was cancelled");
    }
    assert_eq!(
        decode_contract::<Vec<f64>>(PREFIX, DOMAIN, &bytes, LIMITS).unwrap(),
        value
    );
}
#[test]
fn numeric_map_key_grammar_matches_original() {
    for key in [
        "0",
        "-0",
        "01",
        "+1",
        "1 ",
        " 1",
        "1.0",
        "1e2",
        "-3",
        "340282366920938463463374607431768211455",
    ] {
        let value = BTreeMap::from([(key, 1)]);
        let bytes = encode_contract(PREFIX, DOMAIN, &value, LIMITS).unwrap();
        let actual = decode_contract::<BTreeMap<i128, u8>>(PREFIX, DOMAIN, &bytes, LIMITS);
        let reference = original::<BTreeMap<i128, u8>>(&bytes, LIMITS);
        assert_eq!(actual.is_ok(), reference.is_ok(), "key {key}");
        if let (Ok(actual), Ok(reference)) = (actual, reference) {
            assert_eq!(actual, reference);
        }
    }
}
