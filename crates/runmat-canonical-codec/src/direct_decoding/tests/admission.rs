use super::*;
use std::sync::atomic::AtomicBool;

static VISITED: AtomicBool = AtomicBool::new(false);
#[derive(Debug)]
struct Observed;
impl<'de> Deserialize<'de> for Observed {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        VISITED.store(true, Ordering::Relaxed);
        u64::deserialize(deserializer)?;
        Ok(Self)
    }
}

#[test]
fn grammar_precedes_admission_and_admission_precedes_every_typed_visitor() {
    let bytes = encode_contract(PREFIX, DOMAIN, &7u64, LIMITS).unwrap();
    VISITED.store(false, Ordering::Relaxed);
    let result = crate::decode_contract_with_admission::<Observed>(
        PREFIX,
        DOMAIN,
        &bytes,
        LIMITS,
        &NeverCancelled,
        |value| {
            assert_eq!(value, &[7]);
            assert!(!VISITED.load(Ordering::Relaxed));
            Err(CanonicalCodecError::resource_limit(
                "test admission",
                "original rejection",
            ))
        },
    );
    assert_eq!(result.unwrap_err().reason, "original rejection");
    assert!(!VISITED.load(Ordering::Relaxed));
    let mut malformed = bytes.clone();
    malformed.push(0);
    assert!(crate::decode_contract_with_admission::<Observed>(
        PREFIX,
        DOMAIN,
        &malformed,
        LIMITS,
        &NeverCancelled,
        |_| {
            panic!("hostile grammar must fail before admission");
        }
    )
    .is_err());
    assert!(!VISITED.load(Ordering::Relaxed));
    crate::decode_contract_with_admission::<Observed>(
        PREFIX,
        DOMAIN,
        &bytes,
        LIMITS,
        &NeverCancelled,
        |_| Ok(()),
    )
    .unwrap();
    assert!(VISITED.load(Ordering::Relaxed));
}

#[test]
fn nested_exact_vectors_keep_small_and_large_admitted_capacities() {
    let original = vec![Vec::<u64>::new(), vec![3], vec![7; 150_001]];
    let bytes = encode_contract(PREFIX, DOMAIN, &original, LIMITS).unwrap();
    let decoded: crate::ExactVec<crate::ExactVec<u64>> =
        decode_contract(PREFIX, DOMAIN, &bytes, LIMITS).unwrap();
    let outer = decoded.into_vec();
    assert_eq!(outer.capacity(), original.len());
    for (actual, expected) in outer.into_iter().zip(original) {
        let actual = actual.into_vec();
        assert_eq!(actual.capacity(), expected.len());
        assert_eq!(actual, expected);
    }
}

#[test]
fn definite_collection_attribute_preserves_streaming_json_compatibility() {
    #[derive(Debug, Deserialize, PartialEq)]
    struct Row {
        #[serde(deserialize_with = "crate::deserialize_exact_vec")]
        values: Vec<u64>,
    }
    let row: Row = serde_json::from_str("{\"values\":[1,2,3]}").unwrap();
    assert_eq!(row.values, vec![1, 2, 3]);
    let empty: Row = serde_json::from_str("{\"values\":[]}").unwrap();
    assert!(empty.values.is_empty());
    let bytes = encode_contract(
        PREFIX,
        DOMAIN,
        &serde_json::json!({"values": [1,2,3]}),
        LIMITS,
    )
    .unwrap();
    let canonical: Row = decode_contract(PREFIX, DOMAIN, &bytes, LIMITS).unwrap();
    assert_eq!(canonical, row);
    assert_eq!(canonical.values.capacity(), 3);
}
