mod wire_widths;
use std::collections::BTreeMap;

use serde::ser::{SerializeMap, SerializeSeq};

use super::*;

#[test]
fn retained_collection_envelope_and_increments_bound_combined_structural_profile() {
    let limits = CanonicalLimits::new(16 << 20, 100_000_000, 65_536, 64);
    for count in [0, 1, 2, 256] {
        let values: Vec<_> = (0..count)
            .map(|ordinal| format!("original/{ordinal}/Δ/{}", "x".repeat(71)))
            .collect();
        let mut composed = profile_serialized(&Vec::<String>::new(), limits)
            .unwrap()
            .materialization_resident_bytes_estimate;
        for value in &values {
            composed += profile_serialized(&[value], limits)
                .unwrap()
                .materialization_bytes_without_envelope()
                .unwrap();
        }
        let combined = profile_serialized(&values, limits)
            .unwrap()
            .materialization_resident_bytes_estimate;
        assert!(
            composed >= combined,
            "incremental accounting undercounts the original combined contract"
        );
    }
    assert_eq!(
        SerializedCanonicalProfile {
            encoded_bytes_upper_bound: 0,
            materialization_resident_bytes_estimate: 0,
        }
        .materialization_bytes_without_envelope(),
        None
    );
}
use crate::{canonical_payload_materialization_resident_bytes_estimate, encode_contract};

const LIMITS: CanonicalLimits = CanonicalLimits::new(1_000_000, 10_000, 4096, 64);

#[test]
fn combined_profile_visits_a_borrowed_source_once() {
    struct Counted<'a>(&'a std::cell::Cell<usize>);
    impl Serialize for Counted<'_> {
        fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.set(self.0.get() + 1);
            let mut sequence = serializer.serialize_seq(Some(4096))?;
            for _ in 0..4096 {
                sequence.serialize_element(&f64::MAX)?;
            }
            sequence.end()
        }
    }
    let visits = std::cell::Cell::new(0);
    let profile = profile_serialized(&Counted(&visits), LIMITS).unwrap();
    assert_eq!(visits.get(), 1);
    assert_eq!(
        profile.encoded_bytes_upper_bound,
        ENVELOPE_BYTES + 3 + 4096 * 9
    );
    assert!(profile.materialization_resident_bytes_estimate > profile.encoded_bytes_upper_bound);
}

fn covers<T: Serialize>(value: &T) {
    let encoded = encode_contract(b"profile-test\0", "profile", value, LIMITS).unwrap();
    let expected = canonical_payload_materialization_resident_bytes_estimate(&encoded).unwrap();
    let actual = serialized_materialization_resident_bytes_estimate(value, LIMITS).unwrap();
    let profile = profile_serialized(value, LIMITS).unwrap();
    assert!(profile.encoded_bytes_upper_bound >= encoded.len() as u64);
    assert_eq!(profile.materialization_resident_bytes_estimate, actual);
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
            maximum_encoded_bytes: ENVELOPE_BYTES as usize,
            ..LIMITS
        }
    )
    .is_err());
    // Zero uses one CBOR byte; exact scalar admission succeeds, one byte less fails.
    assert!(serialized_materialization_resident_bytes_estimate(
        &0,
        CanonicalLimits {
            maximum_encoded_bytes: ENVELOPE_BYTES as usize + 1,
            ..LIMITS
        }
    )
    .is_ok());
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
        child_peak: 0,
        integer_value_bounds: false,
        live_map_metadata: 0,
        peak_map_metadata: 0,
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
        ..Default::default()
    };
    assert!(counter.string(1, true, 0).is_err());
    counter.profile = MaterializationProfile {
        value_nodes: 0,
        map_entries: 0,
        string_bytes: u64::MAX,
        ..Default::default()
    };
    assert!(counter.string(1, false, 0).is_err());
    assert!(
        materialization_resident_bytes_estimate(MaterializationProfile::default(), u64::MAX)
            .is_err()
    );
}

#[test]
fn json_projection_separates_wire_workspace_and_bounds_materialized_numeric_capacity() {
    let limits = CanonicalLimits::new(16 << 20, 1_000_000, 65_536, 64);
    for count in [0, 1, 32, 4096, 65536] {
        let original = vec![1.25_f64; count];
        let profile = profile_serialized(&original, limits).unwrap();
        let projection = profile.json_projection_resident_bytes_estimate().unwrap();
        let value = serde_json::to_value(&original).unwrap();
        let serde_json::Value::Array(items) = &value else {
            panic!("array projection");
        };
        let requested = (std::mem::size_of::<serde_json::Value>()
            + items.capacity() * std::mem::size_of::<serde_json::Value>())
            as u64;
        assert!(projection >= requested);
        assert_eq!(
            projection + 2 * profile.encoded_bytes_upper_bound,
            profile.materialization_resident_bytes_estimate
        );
    }
    assert!(SerializedCanonicalProfile {
        encoded_bytes_upper_bound: u64::MAX,
        materialization_resident_bytes_estimate: u64::MAX,
    }
    .json_projection_resident_bytes_estimate()
    .is_none());
    assert!(SerializedCanonicalProfile {
        encoded_bytes_upper_bound: 1,
        materialization_resident_bytes_estimate: 1,
    }
    .json_projection_resident_bytes_estimate()
    .is_none());
}

#[test]
fn cooperative_encoding_profile_preserves_counts_and_interrupts_nested_inventory() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Control {
        calls: AtomicUsize,
        maximum: usize,
    }
    impl crate::CanonicalCodecControl for Control {
        fn is_cancelled(&self) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed) >= self.maximum
        }
    }
    let values = vec![vec![1u64, 2, 3, 4]; 1000];
    let limits = crate::CanonicalLimits::new(1024 * 1024, 10000, 1024, 64);
    let complete = Control {
        calls: AtomicUsize::new(0),
        maximum: usize::MAX,
    };
    assert_eq!(
        super::profile_canonical_encoding(&values, limits).unwrap(),
        super::profile_canonical_encoding_with_control(&values, limits, &complete).unwrap()
    );
    assert_eq!(
        super::profile_serialized(&values, limits).unwrap(),
        super::profile_serialized_with_control(&values, limits, &complete).unwrap()
    );
    let cancelled_materialization = Control {
        calls: AtomicUsize::new(0),
        maximum: 50,
    };
    let error = super::profile_serialized_with_control(&values, limits, &cancelled_materialization)
        .unwrap_err();
    assert!(error.to_string().contains("cancel"));
    assert!(cancelled_materialization.calls.load(Ordering::Relaxed) < 100);
    let cancel = Control {
        calls: AtomicUsize::new(0),
        maximum: 50,
    };
    let error =
        super::profile_canonical_encoding_with_control(&values, limits, &cancel).unwrap_err();
    assert!(error.to_string().contains("cancel"));
    assert!(cancel.calls.load(Ordering::Relaxed) < 100);
}
mod shape;

#[test]
fn direct_materialization_stays_within_a_small_factor_of_measured_storage() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Payload {
        result_bytes: Vec<u8>,
        values: Vec<f64>,
        name: String,
    }
    let limits = CanonicalLimits::new(64 << 20, 100_000_000, 65_536, 64);
    let payload = Payload {
        result_bytes: (0..200_000u32).map(|i| (i % 251) as u8).collect(),
        values: (0..20_000).map(|i| i as f64 * 0.5).collect(),
        name: "fluid-state".into(),
    };
    let encoded =
        crate::encode_contract(b"p\0", "direct-materialization", &payload, limits).unwrap();
    let decoded: Payload =
        crate::decode_contract(b"p\0", "direct-materialization", &encoded, limits).unwrap();
    assert_eq!(decoded, payload);
    let profile = profile_direct_materialization(&payload, limits).unwrap();
    assert!(profile.encoded_bytes_upper_bound >= encoded.len() as u64);
    // Covers the import-side typed check over the actual wire.
    let import = crate::canonical_payload_typed_resident_bytes_estimate(&encoded).unwrap();
    assert!(profile.materialization_resident_bytes_estimate >= import);
    // Measured: owned typed storage plus the canonical buffer and the decoded copy.
    let typed =
        payload.result_bytes.capacity() + 8 * payload.values.capacity() + payload.name.len();
    let measured = (typed + encoded.len()) as u64;
    assert!(profile.materialization_resident_bytes_estimate >= measured);
    assert!(
        profile.materialization_resident_bytes_estimate <= 6 * measured,
        "direct estimate {} vs measured {measured}",
        profile.materialization_resident_bytes_estimate
    );
    // The value-tree projection charges tens of bytes per embedded byte.
    let tree = profile_serialized(&payload, limits).unwrap();
    assert!(
        tree.materialization_resident_bytes_estimate
            > 3 * profile.materialization_resident_bytes_estimate
    );
}
