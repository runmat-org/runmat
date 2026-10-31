use runmat_execution::resource::ResourceUsage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SavedResult {
    Success { usage: ResourceUsage },
}

#[test]
fn tagged_accounting_preserves_the_complete_canonical_integer_range() {
    for counter in [0, 1, (1_u128 << 53) + 1, u128::from(u64::MAX), u128::MAX] {
        let original = SavedResult::Success {
            usage: ResourceUsage {
                memory_byte_millis: counter,
                retained_byte_millis: counter,
                ..Default::default()
            },
        };
        let encoded = serde_json::to_vec(&original).unwrap();
        let cold: SavedResult = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(cold, original);
    }
}

#[test]
fn accounting_does_not_coerce_or_round_unadmitted_counters() {
    let encoded = serde_json::to_string(&ResourceUsage::default()).unwrap();
    // Canonical decimal strings are the admitted representation above u64;
    // signed/noncanonical strings, floats and overflowing numbers are not.
    for replacement in [
        "-1",
        "1.0",
        "\"01\"",
        "\"-1\"",
        "18446744073709551616",
        "\"340282366920938463463374607431768211456\"",
    ] {
        let invalid = encoded.replace(
            "\"memory_byte_millis\":0",
            &format!("\"memory_byte_millis\":{replacement}"),
        );
        assert!(serde_json::from_str::<ResourceUsage>(&invalid).is_err());
    }
}
