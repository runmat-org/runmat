//! Compare against the original Serde representation and prove interruption inside conversion.
use super::*;
use crate::{CanonicalCodecControl, NeverCancelled};
use serde::{Deserialize, Serialize};
use std::{
    cell::Cell,
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Motion {
    Unit,
    Scalar(f64),
    Pair(i64, u64),
    Fields {
        label: String,
        optional: Option<Vec<u8>>,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
enum Tagged {
    None,
    Items(Vec<Motion>),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum Alternative {
    Count(u64),
    Text(String),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Contract {
    modes: Vec<Motion>,
    tagged: Tagged,
    alternative: Alternative,
    keys: BTreeMap<i32, String>,
    tuple: (bool, char, i128, u128),
    absent: Option<f64>,
    values: Vec<Option<f64>>,
}
fn fixture() -> Contract {
    Contract {
        modes: vec![
            Motion::Unit,
            Motion::Scalar(-0.0),
            Motion::Pair(i64::MIN, u64::MAX),
            Motion::Fields {
                label: "µ coordinate".into(),
                optional: Some(vec![0, 127, 255]),
            },
        ],
        tagged: Tagged::Items(vec![Motion::Scalar(1.25e-20)]),
        alternative: Alternative::Text("thermal".into()),
        keys: BTreeMap::from([(-4, "negative".into()), (12, "positive".into())]),
        tuple: (true, 'λ', i64::MIN as i128, u64::MAX as u128),
        absent: None,
        values: vec![Some(0.0), Some(1.0e200), None],
    }
}
#[test]
fn typed_projection_and_wire_match_original_serde() {
    let source = fixture();
    let original = serde_json::to_value(&source).unwrap();
    assert_eq!(to_value(&source, &NeverCancelled).unwrap(), original);
    assert_eq!(
        from_value::<Contract>(original.clone(), &NeverCancelled).unwrap(),
        source
    );
    assert_eq!(
        serde_json::from_value::<Contract>(original.clone()).unwrap(),
        source
    );
    let baseline =
        crate::encode_value_envelope(b"prefix\0", "example/v1", &original, &NeverCancelled)
            .unwrap();
    let actual = crate::encode_contract(
        b"prefix\0",
        "example/v1",
        &source,
        crate::CanonicalLimits::new(100_000, 10_000, 10_000, 64),
    )
    .unwrap();
    assert_eq!(actual, baseline);
    assert_eq!(
        crate::decode_contract::<Contract>(
            b"prefix\0",
            "example/v1",
            &actual,
            crate::CanonicalLimits::new(100_000, 10_000, 10_000, 64)
        )
        .unwrap(),
        source
    );
}

struct Bytes(Vec<u8>);
impl Serialize for Bytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}
#[test]
fn byte_slices_and_nonfinite_numbers_keep_original_value_semantics() {
    let bytes = Bytes(vec![0, 1, 255]);
    assert_eq!(
        to_value(&bytes, &NeverCancelled).unwrap(),
        serde_json::to_value(&bytes).unwrap()
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            to_value(&value, &NeverCancelled).unwrap(),
            serde_json::to_value(value).unwrap()
        );
    }
    assert_eq!(
        to_value(&u128::MAX, &NeverCancelled)
            .unwrap_err()
            .to_string(),
        serde_json::to_value(u128::MAX).unwrap_err().to_string()
    );
    let bad = serde_json::json!({"not": "a sequence"});
    assert_eq!(
        from_value::<Vec<u32>>(bad.clone(), &NeverCancelled)
            .unwrap_err()
            .to_string(),
        serde_json::from_value::<Vec<u32>>(bad)
            .unwrap_err()
            .to_string()
    );
}

struct CancelAfter {
    calls: AtomicUsize,
    maximum: usize,
}
impl CanonicalCodecControl for CancelAfter {
    fn is_cancelled(&self) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst) >= self.maximum
    }
}
thread_local! { static CONVERTED: Cell<usize> = const { Cell::new(0) }; }
struct Counted(u32);
impl Serialize for Counted {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CONVERTED.with(|count| count.set(count.get() + 1));
        serializer.serialize_u32(self.0)
    }
}
impl<'de> Deserialize<'de> for Counted {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = u32::deserialize(deserializer)?;
        CONVERTED.with(|count| count.set(count.get() + 1));
        Ok(Self(value))
    }
}
#[test]
fn cancellation_interrupts_typed_serialization_before_materializing_the_inventory() {
    CONVERTED.with(|count| count.set(0));
    let source = (0..1000).map(Counted).collect::<Vec<_>>();
    let control = CancelAfter {
        calls: AtomicUsize::new(0),
        maximum: 25,
    };
    assert!(to_value(&source, &control).is_err());
    let converted = CONVERTED.with(Cell::get);
    assert!(converted > 0 && converted < source.len());
    let bytes = Bytes(vec![1; 1000]);
    let control = CancelAfter {
        calls: AtomicUsize::new(0),
        maximum: 25,
    };
    assert!(to_value(&bytes, &control).is_err());
}
#[test]
fn cancellation_interrupts_typed_deserialization_before_constructing_the_inventory() {
    CONVERTED.with(|count| count.set(0));
    let value = serde_json::to_value((0..1000_u32).collect::<Vec<_>>()).unwrap();
    let control = CancelAfter {
        calls: AtomicUsize::new(0),
        maximum: 25,
    };
    assert!(from_value::<Vec<Counted>>(value, &control).is_err());
    let converted = CONVERTED.with(Cell::get);
    assert!(converted > 0 && converted < 1000);
}
