use std::collections::BTreeMap;

use serde::ser::{SerializeMap, SerializeSeq};

use super::*;
use crate::{canonical_payload_materialization_resident_bytes_estimate, encode_contract};

const LIMITS: CanonicalLimits = CanonicalLimits::new(1_000_000, 10_000, 4096, 64);

fn covers<T: Serialize>(value: &T) {
    let encoded = encode_contract(b"profile-test\0", "profile", value, LIMITS).unwrap();
    let expected = canonical_payload_materialization_resident_bytes_estimate(&encoded).unwrap();
    let actual = serialized_materialization_resident_bytes_estimate(value, LIMITS).unwrap();
    assert!(
        actual >= expected,
        "estimate {actual} < encoded profile {expected}"
    );
}

#[test]
fn primitives_and_collections_cover_encoded_profiles() {
    covers(&());
    covers(&true);
    covers(&(-12_i8, 300_i16, i32::MIN, i64::MIN, -900_i128));
    covers(&(9_u8, 512_u16, u32::MAX, u64::MAX, u64::MAX as u128));
    covers(&(f32::MIN, f64::MAX, -0.0_f64, '猫', "☃"));
    covers(&Option::<String>::None);
    covers(&Some("present"));
    covers(&vec![1.0_f64; 1000]);
    covers(&BTreeMap::from([("a", vec![1, 2]), ("b", vec![3])]));
    covers(&BTreeMap::from([(u128::MAX, false), (0_u128, true)]));
    covers(&BTreeMap::from([(false, 'a'), (true, 'b')]));
    #[derive(Serialize)]
    struct Unit;
    #[derive(Serialize)]
    struct Newtype(Vec<i64>);
    #[derive(Serialize)]
    struct Tuple(i64, &'static str);
    covers(&Unit);
    covers(&Newtype(vec![1, 2]));
    covers(&Tuple(3, "value"));
}

#[test]
fn maximum_canonical_envelope_is_reserved() {
    let mut prefix = vec![b'x'; 128];
    prefix[127] = 0;
    let domain = "x".repeat(128);
    let encoded = encode_contract(&prefix, &domain, &(), LIMITS).unwrap();
    assert!(
        serialized_materialization_resident_bytes_estimate(&(), LIMITS).unwrap()
            >= canonical_payload_materialization_resident_bytes_estimate(&encoded).unwrap()
    );
}

#[test]
fn enum_representations_and_owned_metadata_cover_encoded_profiles() {
    #[derive(Serialize)]
    enum External {
        Unit,
        Newtype(String),
        Tuple(i32, bool),
        Struct { name: String, values: Vec<f64> },
    }
    covers(&External::Unit);
    covers(&External::Newtype("string".into()));
    covers(&External::Tuple(1, true));
    covers(&External::Struct {
        name: "field".into(),
        values: vec![1.0, 2.0],
    });
    #[derive(Serialize)]
    #[serde(tag = "kind")]
    enum Internal {
        Unit,
        Named { label: String },
    }
    covers(&Internal::Unit);
    covers(&Internal::Named { label: "x".into() });
    #[derive(Serialize)]
    #[serde(tag = "kind", content = "value")]
    enum Adjacent {
        Unit,
        Data(Vec<f64>),
    }
    covers(&Adjacent::Unit);
    covers(&Adjacent::Data(vec![4.0]));
    #[derive(Serialize)]
    #[serde(untagged)]
    enum Untagged {
        Number(u32),
        Object { names: Vec<String> },
    }
    covers(&Untagged::Number(7));
    covers(&Untagged::Object {
        names: vec!["a".into(), "b".into()],
    });
    #[derive(Serialize, Ord, PartialOrd, Eq, PartialEq)]
    enum Key {
        Name,
    }
    covers(&BTreeMap::from([(Key::Name, 1)]));
}

#[test]
fn byte_sequences_and_display_are_streamed() {
    struct Bytes;
    impl Serialize for Bytes {
        fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_bytes(&[0, 128, 255])
        }
    }
    covers(&Bytes);
    struct Displayed;
    impl Serialize for Displayed {
        fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_str(&format_args!("{}:{}", "field", 7))
        }
    }
    covers(&Displayed);
    assert!(serialized_materialization_resident_bytes_estimate(
        &Displayed,
        CanonicalLimits {
            maximum_string_bytes: 2,
            ..LIMITS
        }
    )
    .is_err());
}

#[test]
fn known_and_unknown_collection_limits_stop_before_unneeded_values() {
    struct MustNotVisit;
    impl Serialize for MustNotVisit {
        fn serialize<S: ser::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            panic!("over-limit element was traversed")
        }
    }
    assert!(serialized_materialization_resident_bytes_estimate(
        &[MustNotVisit],
        CanonicalLimits {
            maximum_collection_items: 0,
            ..LIMITS
        }
    )
    .is_err());
    struct UnknownLength;
    impl Serialize for UnknownLength {
        fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut seq = serializer.serialize_seq(None)?;
            seq.serialize_element(&0)?;
            seq.serialize_element(&MustNotVisit)?;
            seq.end()
        }
    }
    assert!(serialized_materialization_resident_bytes_estimate(
        &UnknownLength,
        CanonicalLimits {
            maximum_collection_items: 1,
            ..LIMITS
        }
    )
    .is_err());
    #[derive(Serialize)]
    enum Wrapped {
        Value(u32),
    }
    assert!(serialized_materialization_resident_bytes_estimate(
        &Wrapped::Value(1),
        CanonicalLimits {
            maximum_collection_items: 0,
            ..LIMITS
        }
    )
    .is_err());
}

#[test]
fn strings_depth_bytes_and_unsupported_values_fail_closed() {
    let small_string = CanonicalLimits {
        maximum_string_bytes: 2,
        ..LIMITS
    };
    assert!(serialized_materialization_resident_bytes_estimate("three", small_string).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(&'猫', small_string).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(
        &BTreeMap::from([("long", 1)]),
        small_string
    )
    .is_err());
    let shallow = CanonicalLimits {
        maximum_nesting_depth: 0,
        ..LIMITS
    };
    assert!(serialized_materialization_resident_bytes_estimate(&0, shallow).is_ok());
    assert!(serialized_materialization_resident_bytes_estimate(&Vec::<u8>::new(), shallow).is_ok());
    assert!(serialized_materialization_resident_bytes_estimate(&[0], shallow).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(&Some(0), shallow).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(
        &vec![vec![1]],
        CanonicalLimits {
            maximum_nesting_depth: 1,
            ..LIMITS
        }
    )
    .is_err());
    assert!(serialized_materialization_resident_bytes_estimate(
        &0,
        CanonicalLimits {
            maximum_encoded_bytes: ENVELOPE_BYTES as usize + 8,
            ..LIMITS
        }
    )
    .is_err());
    assert!(serialized_materialization_resident_bytes_estimate(&f64::NAN, LIMITS).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(&f64::INFINITY, LIMITS).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(&i128::MIN, LIMITS).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(&u128::MAX, LIMITS).is_err());
    assert!(serialized_materialization_resident_bytes_estimate(
        &BTreeMap::from([(vec![1], 2)]),
        LIMITS
    )
    .is_err());
}

#[test]
fn bad_map_protocol_and_declared_lengths_are_rejected() {
    struct Bad(u8);
    impl Serialize for Bad {
        fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            match self.0 {
                0 => {
                    let mut map = serializer.serialize_map(None)?;
                    map.serialize_value(&1)?;
                    map.end()
                }
                1 => {
                    let mut map = serializer.serialize_map(None)?;
                    map.serialize_key("a")?;
                    map.end()
                }
                2 => {
                    let mut map = serializer.serialize_map(None)?;
                    map.serialize_key("a")?;
                    map.serialize_key("b")?;
                    map.end()
                }
                3 => serializer.serialize_seq(Some(1))?.end(),
                4 => {
                    let mut seq = serializer.serialize_seq(Some(0))?;
                    seq.serialize_element(&1)?;
                    seq.end()
                }
                _ => Err(ser::Error::custom("caller failure")),
            }
        }
    }
    for case in 0..6 {
        assert!(serialized_materialization_resident_bytes_estimate(&Bad(case), LIMITS).is_err());
    }
}

#[test]
fn all_counter_arithmetic_is_checked() {
    let mut counter = Counter {
        profile: MaterializationProfile::default(),
        encoded_bytes: u64::MAX,
        limits: LIMITS,
    };
    assert!(counter.encoded(1).is_err());
    counter.encoded_bytes = 0;
    counter.profile.value_nodes = u64::MAX;
    assert!(counter.node(0).is_err());
    counter.profile = MaterializationProfile {
        value_nodes: 0,
        map_entries: u64::MAX,
        string_bytes: 0,
    };
    assert!(counter.string(1, true, 0).is_err());
    counter.profile = MaterializationProfile {
        value_nodes: 0,
        map_entries: 0,
        string_bytes: u64::MAX,
    };
    assert!(counter.string(1, false, 0).is_err());
    assert!(
        materialization_resident_bytes_estimate(MaterializationProfile::default(), u64::MAX)
            .is_err()
    );
}
