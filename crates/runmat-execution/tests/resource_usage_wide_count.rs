//! Resource accounting must survive JSON inside internally tagged enums.
use runmat_execution::resource::ResourceUsage;

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
#[serde(tag = "kind")]
enum Tagged {
    Values { usage: ResourceUsage },
}

#[test]
fn counts_round_trip_inside_tagged_enums_without_changing_narrow_encoding() {
    for count in [
        0_u128,
        5,
        u128::from(u64::MAX),
        u128::from(u64::MAX) + 1,
        u128::MAX,
    ] {
        let usage = ResourceUsage {
            memory_byte_millis: count,
            retained_byte_millis: count,
            ..Default::default()
        };
        let plain = serde_json::to_string(&usage).unwrap();
        if count <= u128::from(u64::MAX) {
            assert!(plain.contains(&format!("\"memory_byte_millis\":{count}")));
        }
        assert_eq!(
            serde_json::from_str::<ResourceUsage>(&plain).unwrap(),
            usage
        );
        let tagged = Tagged::Values { usage };
        let encoded = serde_json::to_string(&tagged).unwrap();
        assert_eq!(serde_json::from_str::<Tagged>(&encoded).unwrap(), tagged);
    }
}
