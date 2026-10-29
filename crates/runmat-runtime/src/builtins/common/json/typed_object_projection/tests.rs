use super::*;
use serde_json::json;

#[test]
fn sparse_null_provenance_restores_nested_enum_maps_and_array_slots() {
    let original = json!({"steps":[{"Completed":{
        "maximum_resident_bytes":null,
        "slots":[null,[null,[],9007199254740993u64],{"x":null}]
    }}],"numeric":[1,2,3]});
    let mut projected = original.clone();
    let shape = Projection::capture(&mut projected).unwrap();
    let metadata = shape.to_json().unwrap();
    assert!(!metadata.to_string().contains("numeric"));
    let cold = Projection::from_json(metadata).unwrap();
    cold.restore(&mut projected).unwrap();
    assert_eq!(projected, original);
}

#[test]
fn malformed_unknown_and_conflicting_provenance_is_rejected() {
    for metadata in [
        json!({"version":2,"root":{"kind":"Null"}}),
        json!({"version":1,"root":{"kind":"Null"},"extra":1}),
        json!({"version":1,"root":{"kind":"Unknown"}}),
        json!({"version":1,"root":{"kind":"Object","fields":{}}}),
        json!({"version":1,"root":{"kind":"Array","length":1,
            "elements":[{"index":1,"shape":{"kind":"Null"}}]}}),
        json!({"version":1,"root":{"kind":"Array","length":1,
            "elements":[{"index":0,"shape":{"kind":"Null"}},
                        {"index":0,"shape":{"kind":"Null"}}]}}),
    ] {
        assert!(Projection::from_json(metadata).is_err());
    }
    let mut json = json!({"x":null});
    let shape = Projection::capture(&mut json).unwrap();
    json["x"] = json!(7);
    assert!(shape.restore(&mut json).is_err());
}
