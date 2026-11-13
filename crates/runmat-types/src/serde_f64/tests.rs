use super::*;
use serde::Serialize;

#[derive(Debug, Serialize, Deserialize)]
struct Float(#[serde(with = "super")] f64);

#[derive(Debug, Serialize, Deserialize)]
struct Floats(#[serde(with = "super::vec")] Vec<f64>);

#[test]
fn finite_float_payloads_are_byte_identical_to_existing_serde_numbers() {
    for value in [
        0.0,
        -0.0,
        1.0,
        -42.25,
        f64::EPSILON,
        f64::MIN_POSITIVE,
        f64::MAX,
    ] {
        let encoded = serde_json::to_string(&Float(value)).expect("finite float");
        assert_eq!(encoded, serde_json::to_string(&value).unwrap());
        let decoded: Float = serde_json::from_str(&encoded).expect("finite roundtrip");
        assert_eq!(decoded.0.to_bits(), value.to_bits(), "{encoded}");
    }
    assert_eq!(
        serde_json::from_str::<Float>("-0.0").unwrap().0.to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[test]
fn nonfinite_tags_are_deterministic_and_preserve_every_payload_bit() {
    for bits in [
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0000,
        0xfff8_0000_0000_0042,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
        0x7fff_ffff_ffff_ffff,
    ] {
        let value = Float(f64::from_bits(bits));
        let encoded = serde_json::to_string(&value).unwrap();
        assert_eq!(encoded, format!("{{\"f64_bits\":\"{bits:016x}\"}}"));
        assert_eq!(serde_json::to_string(&value).unwrap(), encoded);
        let decoded: Float = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.0.to_bits(), bits);
    }
}

#[test]
fn float_vectors_keep_finite_numbers_and_nonfinite_bits() {
    let values = vec![
        1.5,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0042),
    ];
    let encoded = serde_json::to_string(&Floats(values.clone())).unwrap();
    assert!(encoded.starts_with("[1.5,-0.0,{\"f64_bits\":"));
    let decoded: Floats = serde_json::from_str(&encoded).unwrap();
    for (actual, expected) in decoded.0.iter().zip(values) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert!(serde_json::from_str::<Floats>("[1.0,null]").is_err());
}

#[test]
fn malformed_null_ambiguous_and_noncanonical_tags_are_rejected() {
    for encoded in [
        "null",
        "true",
        "[]",
        "{}",
        "\"NaN\"",
        r#"{"other":"7ff0000000000000"}"#,
        r#"{"f64_bits":"7ff0000000000000","other":0}"#,
        r#"{"f64_bits":"7ff0000000000000","f64_bits":"fff0000000000000"}"#,
        r#"{"f64_bits":null}"#,
        r#"{"f64_bits":9218868437227405312}"#,
        r#"{"f64_bits":"7ff0"}"#,
        r#"{"f64_bits":"7FF0000000000000"}"#,
        r#"{"f64_bits":"0x7ff0000000000000"}"#,
        r#"{"f64_bits":"7ff000000000000g"}"#,
        r#"{"f64_bits":"3ff0000000000000"}"#,
        r#"{"f64_bits":"8000000000000000"}"#,
    ] {
        assert!(
            serde_json::from_str::<Float>(encoded).is_err(),
            "accepted {encoded}"
        );
    }
}

#[test]
fn argument_declarations_keep_finite_fixtures_and_roundtrip_nonfinite_literals() {
    use crate::{FunctionArgDefaultValue, FunctionArgValidationLiteral, FunctionArgValidator};
    assert_eq!(
        serde_json::to_string(&FunctionArgDefaultValue::Number(1.5)).unwrap(),
        r#"{"Number":1.5}"#
    );
    assert_eq!(
        serde_json::to_string(&FunctionArgValidationLiteral::Number(-0.0)).unwrap(),
        r#"{"Number":-0.0}"#
    );
    for bits in [
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        0xfff8_0000_0000_0042,
    ] {
        let original = FunctionArgDefaultValue::Number(f64::from_bits(bits));
        let json = serde_json::to_string(&original).unwrap();
        let FunctionArgDefaultValue::Number(value) = serde_json::from_str(&json).unwrap() else {
            panic!("number variant");
        };
        assert_eq!(value.to_bits(), bits);
        let original = FunctionArgValidationLiteral::Number(f64::from_bits(bits));
        let json = serde_json::to_string(&original).unwrap();
        let FunctionArgValidationLiteral::Number(value) = serde_json::from_str(&json).unwrap()
        else {
            panic!("number variant");
        };
        assert_eq!(value.to_bits(), bits);
    }
    let bound = FunctionArgValidator::GreaterThan(f64::NEG_INFINITY);
    let json = serde_json::to_string(&bound).unwrap();
    let FunctionArgValidator::GreaterThan(value) = serde_json::from_str(&json).unwrap() else {
        panic!("bound variant");
    };
    assert_eq!(value.to_bits(), f64::NEG_INFINITY.to_bits());
}
