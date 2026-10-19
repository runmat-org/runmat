use super::*;
const LIMITS: CanonicalLimits = CanonicalLimits::new(1 << 20, 1024, 4096, 16);

#[test]
fn structural_density_changes_admission_without_a_blanket_wire_multiplier() {
    let dense = serde_json::to_vec(&vec![0; 1000]).unwrap();
    let text = serde_json::to_vec(&"x".repeat(2000)).unwrap();
    let dense_size = json_materialization_resident_bytes_estimate(&dense, LIMITS, 1 << 20).unwrap();
    let text_size = json_materialization_resident_bytes_estimate(&text, LIMITS, 1 << 20).unwrap();
    assert!(dense_size > text_size);
    assert!(json_materialization_resident_bytes_estimate(&dense, LIMITS, text_size).is_err());
    assert_eq!(
        json_materialization_resident_bytes_estimate(&text, LIMITS, text_size).unwrap(),
        text_size
    );
}

#[test]
fn exact_profile_ceiling_and_escaped_utf8_preserve_json_semantics() {
    let bytes = br#"{"key\n":"\u03bb","array":[true,null,-3,1.25]}"#;
    let estimate = json_materialization_resident_bytes_estimate(bytes, LIMITS, 1 << 20).unwrap();
    assert!(json_materialization_resident_bytes_estimate(bytes, LIMITS, estimate - 1).is_err());
    assert_eq!(
        json_materialization_resident_bytes_estimate(bytes, LIMITS, estimate).unwrap(),
        estimate
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(bytes).unwrap()["key\n"],
        "λ"
    );
}

#[test]
fn malformed_trailing_and_excessive_structure_reject_without_materialization() {
    for bytes in [
        b"[] {}".as_slice(),
        b"{\"a\":}".as_slice(),
        b"[1,2,3]".as_slice(),
    ] {
        let small = CanonicalLimits::new(4096, 2, 16, 2);
        assert!(json_materialization_resident_bytes_estimate(bytes, small, 1 << 20).is_err());
    }
    let nested = b"[[[[0]]]]";
    assert!(json_materialization_resident_bytes_estimate(
        nested,
        CanonicalLimits::new(4096, 8, 16, 2),
        1 << 20
    )
    .is_err());
    assert!(json_materialization_resident_bytes_estimate(
        br#""long string""#,
        CanonicalLimits::new(4096, 8, 2, 2),
        1 << 20
    )
    .is_err());
}

#[test]
fn structured_limits_and_malformed_json_have_distinct_causes() {
    use crate::CanonicalCodecErrorKind::{Invalid, ResourceLimit};
    let valid = br#"{"original":[1,2,3],"text":"abcdef"}"#;
    let required = json_materialization_resident_bytes_estimate(valid, LIMITS, 1 << 20).unwrap();
    assert_eq!(
        json_materialization_resident_bytes_estimate(valid, LIMITS, required - 1)
            .unwrap_err()
            .kind,
        ResourceLimit
    );
    for limits in [
        CanonicalLimits::new(1, 1024, 4096, 16),
        CanonicalLimits::new(4096, 2, 4096, 16),
        CanonicalLimits::new(4096, 1024, 2, 16),
        CanonicalLimits::new(4096, 1024, 4096, 0),
    ] {
        assert_eq!(
            json_materialization_resident_bytes_estimate(valid, limits, 1 << 20)
                .unwrap_err()
                .kind,
            ResourceLimit
        );
    }
    for malformed in [
        br#"{"original":}"#.as_slice(),
        br#"[] {}"#.as_slice(),
        br#""bad\q""#.as_slice(),
    ] {
        assert_eq!(
            json_materialization_resident_bytes_estimate(malformed, LIMITS, 1 << 20)
                .unwrap_err()
                .kind,
            Invalid
        );
    }
    assert_eq!(
        json_materialization_resident_bytes_estimate(valid, LIMITS, required).unwrap(),
        required
    );
}
