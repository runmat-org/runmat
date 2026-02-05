use std::fs;
use std::path::PathBuf;

#[test]
fn public_api_spec_parses() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../openapi/runmat-public.yaml");
    let contents = fs::read_to_string(&path).expect("read spec");
    let spec: openapiv3::OpenAPI = serde_yaml::from_str(&contents).expect("parse spec");
    assert_eq!(spec.openapi, "3.0.3");
}
