use super::*;
use crate::NeverCancelled;
use serde::ser::{SerializeMap, SerializeSeq};
use std::collections::BTreeMap;
const LIMITS: CanonicalLimits = CanonicalLimits::new(16 << 20, 1_000_000, 65_536, 64);
fn original<T: Serialize>(value: &T) -> Vec<u8> {
    let value = crate::controlled_serde::to_value(value, &NeverCancelled).unwrap();
    let mut bytes = b"test\0".to_vec();
    let mut encoder = minicbor::Encoder::new(&mut bytes);
    encoder.array(2).unwrap().str("direct").unwrap();
    crate::value::encode_value(&mut encoder, &value, &NeverCancelled).unwrap();
    bytes
}
fn equivalent<T: Serialize>(value: &T) {
    let encoded = encode(b"test\0", "direct", value, LIMITS, &NeverCancelled).unwrap();
    assert_eq!(encoded, original(value));
    assert_eq!(streamed(value), encoded);
}
/// The streamed encoding, collected chunk by chunk.
fn streamed<T: Serialize>(value: &T) -> Vec<u8> {
    let mut collected = Vec::new();
    let mut chunks = 0usize;
    let count = stream(
        b"test\0",
        "direct",
        value,
        LIMITS,
        &NeverCancelled,
        &mut |chunk: &[u8]| {
            assert!(!chunk.is_empty());
            chunks += 1;
            collected.extend_from_slice(chunk);
        },
    )
    .unwrap();
    assert_eq!(count, collected.len() as u64);
    assert!(chunks >= 1);
    collected
}
#[derive(Serialize)]
enum Variant {
    Unit,
    Value(f32),
    Tuple(bool, u64),
    Record { z: Vec<f64>, a: Option<String> },
}
#[test]
fn established_scalar_container_enum_and_duplicate_key_bytes_are_preserved() {
    equivalent(&(i64::MIN, u64::MAX, 'λ', -0.0_f64, f64::NAN));
    equivalent(&[0.1_f32, f32::INFINITY]);
    equivalent(&Variant::Unit);
    equivalent(&Variant::Value(0.1));
    equivalent(&Variant::Tuple(true, u64::MAX));
    equivalent(&Variant::Record {
        z: vec![1.25; 4096],
        a: Some("Δ".into()),
    });
    equivalent(&BTreeMap::from([(true, "yes"), (false, "no")]));
    struct Repeated;
    impl Serialize for Repeated {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(None)?;
            map.serialize_entry("z", &[1, 2, 3])?;
            map.serialize_entry("a", &42)?;
            map.serialize_entry("z", &[9])?;
            map.end()
        }
    }
    equivalent(&Repeated);
}
#[test]
fn unknown_sequences_preserve_original_bytes_and_enforce_wire_ceiling() {
    struct Unknown;
    impl Serialize for Unknown {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut sequence = serializer.serialize_seq(None)?;
            for value in 0..32 {
                sequence.serialize_element(&value)?;
            }
            sequence.end()
        }
    }
    equivalent(&Unknown);
    let encoded = original(&Unknown);
    let exact = CanonicalLimits::new(encoded.len(), 32, 65_536, 64);
    assert_eq!(
        encode(b"test\0", "direct", &Unknown, exact, &NeverCancelled).unwrap(),
        encoded
    );
    assert!(encode(
        b"test\0",
        "direct",
        &Unknown,
        CanonicalLimits::new(encoded.len() - 1, 32, 65_536, 64),
        &NeverCancelled
    )
    .is_err());
    assert_eq!(
        encode(
            b"test\0",
            "direct",
            &Unknown,
            CanonicalLimits::new(4096, 31, 65_536, 64),
            &NeverCancelled
        )
        .unwrap(),
        encoded
    );
}
#[test]
fn cancellation_interrupts_original_nested_inventory_and_overflow_rejects() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Cancel(AtomicUsize);
    impl CanonicalCodecControl for Cancel {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::SeqCst) >= 20
        }
    }
    let value = vec![vec![1.25_f64; 4096]; 4];
    assert!(encode(
        b"test\0",
        "direct",
        &value,
        LIMITS,
        &Cancel(AtomicUsize::new(0))
    )
    .is_err());
    equivalent(&value);
    assert!(encode(b"test\0", "direct", &u128::MAX, LIMITS, &NeverCancelled).is_err());
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Numeric array fixtures bound original encoded bytes and distinguish direct
//   byte-buffer workspace from JSON materialization. This is a requested component estimate, not
//   measured allocator usage, RSS, whole-task peak or complete-study scalability qualification.
#[test]
fn direct_encoding_profile_bounds_original_wire_and_numeric_workspace_scales_with_bytes() {
    let original = vec![1.25_f64; 65536];
    let profile = crate::profile_canonical_encoding(&original, LIMITS).unwrap();
    let materialization = crate::profile_serialized(&original, LIMITS).unwrap();
    let wire = encode(b"test\0", "direct", &original, LIMITS, &NeverCancelled).unwrap();
    assert!(profile.encoded_bytes_upper_bound >= wire.len() as u64);
    // The output is reserved once at the profiled bound and the sequence streams into it.
    assert_eq!(profile.encoded_bytes_upper_bound, wire.capacity() as u64);
    assert_eq!(
        profile.resident_workspace_bytes_estimate,
        profile.encoded_bytes_upper_bound
            + crate::materialization::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT
    );
    assert!(
        profile.resident_workspace_bytes_estimate
            < materialization.materialization_resident_bytes_estimate
    );
}

#[test]
fn trusted_encoding_preserves_original_projection_while_decode_and_profiles_bound_shape() {
    // Existing identity-only contracts use shape limits below the trusted serde
    // representation (for example a 71-byte tagged digest with a 64-byte limit).
    // Changing encoding admission during an allocation optimization breaks these
    // identities. Hostile decode and preallocation profiles still enforce shape.
    let value = vec![vec!["sha256:".to_owned() + &"0".repeat(64)]; 3];
    let limits = CanonicalLimits::new(4096, 2, 64, 0);
    let bytes = crate::encode_contract(b"test\0", "direct", &value, limits).unwrap();
    assert_eq!(bytes, original(&value));
    assert!(
        crate::decode_contract::<Vec<Vec<String>>>(b"test\0", "direct", &bytes, limits).is_err()
    );
    assert!(crate::profile_serialized(&value, limits).is_err());
    assert!(crate::profile_canonical_encoding(&value, limits).is_err());
    assert_eq!(value[0][0].len(), 71);
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Sibling, wide and nested maps exercise live sorted-map metadata and preserve original
//   canonical bytes. Component demand only; no measured allocator, whole-task peak or
//   complete-study claim.
#[test]
fn encoding_workspace_retains_ancestor_maps_and_releases_sibling_metadata() {
    let children = vec![std::collections::BTreeMap::from([("key", "borrowed".repeat(128))]); 1024];
    let repeated = crate::profile_canonical_encoding(&children, LIMITS).unwrap();
    let single = crate::profile_canonical_encoding(&children[..1], LIMITS).unwrap();
    let overhead = |profile: crate::CanonicalEncodingProfile| {
        profile.resident_workspace_bytes_estimate - profile.encoded_bytes_upper_bound
    };
    assert_eq!(overhead(repeated), overhead(single));
    let wide =
        std::collections::BTreeMap::from_iter((0..1024).map(|i| (format!("key{i}"), "borrowed")));
    let wide = crate::profile_canonical_encoding(&wide, LIMITS).unwrap();
    assert!(overhead(wide) > overhead(single) + 1024 * 160);
    let nested = std::collections::BTreeMap::from([("outer", &children)]);
    let nested = crate::profile_canonical_encoding(&nested, LIMITS).unwrap();
    assert!(overhead(nested) > overhead(single));
    let bytes = encode(b"test\0", "direct", &children, LIMITS, &NeverCancelled).unwrap();
    assert_eq!(bytes, original(&children));
    assert!(repeated.encoded_bytes_upper_bound >= bytes.len() as u64);
}

#[test]
fn profiled_output_reservation_preserves_bytes_without_retained_growth_slack() {
    let value = BTreeMap::from([
        ("values", vec![1.25f64; 32_768]),
        ("other", vec![-0.0f64; 1024]),
    ]);
    let profile = crate::profile_canonical_encoding(&value, LIMITS).unwrap();
    let capacity = profile.encoded_bytes_upper_bound as usize;
    let reserved = crate::encode_contract_with_capacity(
        b"test\0",
        "direct",
        &value,
        LIMITS,
        &NeverCancelled,
        capacity,
    )
    .unwrap();
    assert_eq!(reserved, original(&value));
    assert_eq!(reserved.capacity(), capacity);
    assert!(crate::encode_contract_with_capacity(
        b"test\0",
        "direct",
        &value,
        LIMITS,
        &NeverCancelled,
        LIMITS.maximum_encoded_bytes + 1
    )
    .is_err());
}

#[derive(Serialize)]
struct Declared {
    zeta: u8,
    alpha: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    middle: Option<Box<Declared>>,
    beta: Variant,
    #[serde(flatten)]
    extra: BTreeMap<String, u8>,
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Tagged {
    First { zz: u8, aa: Vec<f64> },
    Second(Declared),
}
#[derive(Serialize)]
struct Ordered {
    a: u8,
    b: Vec<u16>,
    c: Option<Ordered2>,
}
#[derive(Serialize)]
struct Ordered2 {
    x: String,
    y: [u8; 3],
}

#[test]
fn streamed_and_held_struct_fields_preserve_canonical_bytes() {
    let leaf = Declared {
        zeta: 1,
        alpha: vec![3, 2, 1],
        middle: None,
        beta: Variant::Record {
            z: vec![0.5],
            a: None,
        },
        extra: BTreeMap::new(),
    };
    let nested = Declared {
        zeta: 9,
        alpha: (0..300).collect(),
        middle: Some(Box::new(leaf)),
        beta: Variant::Tuple(false, 7),
        extra: BTreeMap::from([("aa".into(), 1), ("zz".into(), 2)]),
    };
    equivalent(&nested);
    equivalent(&Tagged::First {
        zz: 3,
        aa: vec![1.0, -2.0],
    });
    equivalent(&Tagged::Second(nested));
    equivalent(&Ordered {
        a: 1,
        b: vec![1, 2],
        c: Some(Ordered2 {
            x: "x".into(),
            y: [1, 2, 3],
        }),
    });
    equivalent(&Vec::<Ordered>::new());
}

#[test]
fn declared_maps_fall_back_from_streaming_without_changing_bytes() {
    struct Keys(&'static [(&'static str, u32)], Option<usize>);
    impl Serialize for Keys {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(self.1)?;
            for (key, value) in self.0 {
                map.serialize_entry(key, &vec![*value; *value as usize])?;
            }
            map.end()
        }
    }
    // Streams throughout.
    equivalent(&Keys(&[("a", 1), ("b", 2), ("c", 3)], Some(3)));
    // Out of order after streaming two entries.
    equivalent(&Keys(&[("a", 1), ("c", 3), ("b", 2), ("d", 4)], Some(4)));
    // A repeated key keeps the last value, as the established projection does.
    equivalent(&Keys(&[("a", 1), ("b", 2), ("b", 5)], Some(3)));
    // A declared length that differs from the entries.
    equivalent(&Keys(&[("a", 1), ("b", 2)], Some(5)));
    // Unknown length.
    equivalent(&Keys(&[("b", 1), ("a", 2)], None));
    // Nested inside a held struct field.
    equivalent(&BTreeMap::from([(
        "outer",
        vec![Keys(&[("x", 1), ("w", 2)], Some(2))],
    )]));
}

// A digest-only encoding hands out the exact canonical bytes while the root buffer drains:
// declared-length maps that later fall back to sorted holding rewrite only bytes after their
// own pinned header, struct fields stream in canonical order, and nested maps unpin in order.
#[test]
fn streamed_encoding_is_byte_identical_while_the_root_buffer_drains() {
    #[derive(Serialize)]
    struct Wide {
        zeta: Vec<u64>,
        alpha: BTreeMap<String, Vec<f64>>,
        mid: Vec<Variant>,
    }
    struct Unsorted(Vec<(&'static str, Vec<u32>)>);
    impl Serialize for Unsorted {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let mut map = s.serialize_map(Some(self.0.len()))?;
            for (key, value) in &self.0 {
                map.serialize_entry(key, value)?;
            }
            map.end()
        }
    }
    equivalent(&Wide {
        zeta: (0..5000).collect(),
        alpha: (0..40)
            .map(|i| (format!("k{i:03}"), vec![i as f64; 100]))
            .collect(),
        mid: (0..300)
            .map(|i| Variant::Record {
                z: vec![i as f64; 7],
                a: Some(i.to_string()),
            })
            .collect(),
    });
    equivalent(&Unsorted(vec![
        ("b", (0..500).collect()),
        ("c", (0..500).collect()),
        ("a", (0..500).collect()),
    ]));
    equivalent(&vec![
        Unsorted(vec![("x", vec![1; 300]), ("y", vec![2; 300])]),
        Unsorted(vec![("y", vec![3; 300]), ("x", vec![4; 300])]),
    ]);
    // A streamed encoding obeys the same byte ceiling as the materialized one.
    let tight = CanonicalLimits::new(64, 1_000_000, 65_536, 64);
    let error = stream(
        b"test\0",
        "direct",
        &vec![0u64; 100],
        tight,
        &NeverCancelled,
        &mut |_: &[u8]| {},
    )
    .unwrap_err();
    assert_eq!(error.kind, crate::CanonicalCodecErrorKind::ResourceLimit);
}
