use super::*;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

#[test]
fn inspect_detects_stl() {
    let result =
        geometry_inspect("/part.stl", TRIANGLE_STL.as_bytes()).expect("inspect should work");
    assert_eq!(result.format, "stl");
}

#[test]
fn inspect_op_returns_typed_metadata() {
    let envelope = geometry_inspect_op(
        "/part.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-g1".to_string()), Some("request-g1".to_string())),
    )
    .expect("inspect envelope should work");

    assert_eq!(envelope.operation, "geometry.inspect");
    assert_eq!(envelope.op_version, "geometry.inspect/v1");
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-g1"));
    assert_eq!(envelope.data.format, "stl");
}

#[test]
fn load_and_stats_work_for_stl() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let stats = geometry_compute_stats(&asset).expect("stats should work");
    assert_eq!(stats.mesh_count, 1);
    assert_eq!(stats.total_elements, 1);
}

#[test]
fn inspect_and_load_step_work() {
    let inspect =
        geometry_inspect("/assembly.step", SIMPLE_STEP.as_bytes()).expect("inspect should work");
    assert_eq!(inspect.format, "step");

    let asset = geometry_load("/assembly.step", SIMPLE_STEP.as_bytes()).expect("load should work");
    assert_eq!(asset.source.importer_version, "step/v1");
    assert_eq!(asset.regions.len(), 1);
    assert!(asset.source_geometry.assembly.is_some());
}

#[test]
fn load_op_maps_unsupported_format_error_code() {
    let error = geometry_load_op(
        "/bad.bin",
        b"not geometry",
        OperationContext::new(None, None),
    )
    .expect_err("load should fail");

    assert_eq!(error.error_code, "GEOMETRY_FORMAT_UNSUPPORTED");
    assert_eq!(error.operation, "geometry.load");
    assert_eq!(error.op_version, "geometry.load/v1");
}
