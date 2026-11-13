use super::*;

#[derive(Debug, Serialize, Deserialize)]
struct Float(#[serde(with = "super")] f32);

#[derive(Debug, Serialize, Deserialize)]
struct Floats(#[serde(with = "super::vec")] Vec<f32>);

#[test]
fn finite_single_values_keep_json_numbers_and_signed_zero() {
    for value in [0.0_f32, -0.0, 1.0, -42.25, f32::EPSILON, f32::MAX] {
        let encoded = serde_json::to_string(&Float(value)).unwrap();
        assert_eq!(encoded, serde_json::to_string(&value).unwrap());
        let decoded: Float = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.0.to_bits(), value.to_bits());
    }
}

#[test]
fn nonfinite_single_tags_and_vectors_preserve_ieee_bits() {
    let bits = [
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0042,
        0xffc0_0001,
        0x7f80_0001,
    ];
    let values = bits.map(f32::from_bits);
    for (bit, value) in bits.iter().copied().zip(values.iter().copied()) {
        let encoded = serde_json::to_string(&Float(value)).unwrap();
        assert_eq!(encoded, format!("{{\"f32_bits\":\"{bit:08x}\"}}"));
        let decoded: Float = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.0.to_bits(), bit);
    }
    let mut vector = vec![1.5, -0.0];
    vector.extend(values);
    let encoded = serde_json::to_string(&Floats(vector.clone())).unwrap();
    assert!(encoded.starts_with("[1.5,-0.0,{\"f32_bits\":"));
    let decoded: Floats = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.0.len(), vector.len());
    for (actual, expected) in decoded.0.iter().zip(vector) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn ambiguous_null_and_noncanonical_single_tags_reject() {
    for encoded in [
        "null",
        r#"{"f32_bits":"7f800000","other":0}"#,
        r#"{"f32_bits":"7F800000"}"#,
        r#"{"f32_bits":"0x7f800000"}"#,
        r#"{"f32_bits":"3f800000"}"#,
        r#"{"f64_bits":"7ff0000000000000"}"#,
    ] {
        assert!(
            serde_json::from_str::<Float>(encoded).is_err(),
            "accepted {encoded}"
        );
    }
    assert!(serde_json::from_str::<Floats>("[1.0,null]").is_err());
}
