use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    #[serde(with = "crate::wire_integer")]
    count: usize,
    #[serde(with = "crate::wire_integer")]
    signed: i64,
    #[serde(with = "crate::wire_integer::option")]
    limit: Option<u64>,
    #[serde(with = "crate::wire_integer::vec")]
    extents: Vec<u64>,
}

fn decode(json: &str) -> Result<Probe, serde_json::Error> {
    serde_json::from_str(json)
}

#[test]
fn bound_is_javascript_safe_integer() {
    assert_eq!(MAX_WIRE_INTEGER, 9_007_199_254_740_991);
    assert_eq!(bounded(MAX_WIRE_INTEGER), Some(MAX_WIRE_INTEGER));
    assert_eq!(bounded(MAX_WIRE_INTEGER + 1), None);
    assert_eq!(bounded(u64::MAX), None);
    assert_eq!(
        bounded(-(MAX_WIRE_INTEGER as i64)),
        Some(-(MAX_WIRE_INTEGER as i64))
    );
    assert_eq!(bounded(-(MAX_WIRE_INTEGER as i64) - 1), None);
    assert_eq!(
        bounded_whole_f64(9_007_199_254_740_991.0),
        Some(MAX_WIRE_INTEGER)
    );
    assert_eq!(bounded_whole_f64(9_007_199_254_740_992.0), None);
    assert_eq!(bounded_whole_f64(1.5), None);
    assert_eq!(bounded_whole_f64(-1.0), None);
}

#[test]
fn decode_accepts_boundary_and_null_and_rejects_above() {
    let max = r#"{"count":9007199254740991,"signed":-9007199254740991,"limit":9007199254740991,"extents":[0,9007199254740991]}"#;
    let probe = decode(max).expect("2^53-1 is admitted");
    assert_eq!(probe.limit, Some(MAX_WIRE_INTEGER));
    assert_eq!(serde_json::to_string(&probe).unwrap(), max);

    let null = decode(r#"{"count":0,"signed":0,"limit":null,"extents":[]}"#).expect("null");
    assert_eq!(null.limit, None);
    assert_eq!(
        serde_json::to_string(&null).unwrap(),
        r#"{"count":0,"signed":0,"limit":null,"extents":[]}"#
    );

    for rejected in [
        r#"{"count":9007199254740992,"signed":0,"limit":null,"extents":[]}"#,
        r#"{"count":0,"signed":-9007199254740992,"limit":null,"extents":[]}"#,
        r#"{"count":0,"signed":0,"limit":9007199254740992,"extents":[]}"#,
        r#"{"count":0,"signed":0,"limit":18446744073709551615,"extents":[]}"#,
        r#"{"count":0,"signed":0,"limit":null,"extents":[9007199254740992]}"#,
        r#"{"count":0,"signed":0,"limit":-1,"extents":[]}"#,
        r#"{"count":0,"signed":0,"limit":1.5,"extents":[]}"#,
        r#"{"count":-1,"signed":0,"limit":null,"extents":[]}"#,
        r#"{"count":0,"signed":0,"extents":[]}"#,
    ] {
        assert!(decode(rejected).is_err(), "{rejected} must be rejected");
    }
}

#[test]
fn encode_rejects_values_outside_the_wire_range() {
    let probe = Probe {
        count: 0,
        signed: 0,
        limit: Some(u64::MAX),
        extents: Vec::new(),
    };
    assert!(serde_json::to_string(&probe).is_err());
    let probe = Probe {
        count: 0,
        signed: 0,
        limit: None,
        extents: vec![MAX_WIRE_INTEGER + 1],
    };
    assert!(serde_json::to_string(&probe).is_err());
}
