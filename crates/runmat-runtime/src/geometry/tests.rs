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

#[test]
fn list_regions_op_returns_versioned_envelope() {
    let asset = geometry_load("/assembly.step", SIMPLE_STEP.as_bytes()).expect("load should work");
    let envelope = geometry_list_regions_op(
        &asset,
        OperationContext::new(Some("trace-g2".to_string()), None),
    )
    .expect("list regions should work");

    assert_eq!(envelope.operation, "geometry.list_regions");
    assert_eq!(envelope.op_version, "geometry.list_regions/v1");
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-g2"));
    assert_eq!(envelope.data.regions.len(), 1);
}

#[test]
fn query_entities_op_returns_deterministic_nodes() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let envelope = geometry_query_entities_op(
        &asset,
        GeometryEntityQuery {
            region_id: None,
            mesh_id: None,
            entity_kind: EntityKind::Node,
            limit: Some(2),
        },
        OperationContext::new(Some("trace-g3".to_string()), None),
    )
    .expect("query should work");

    assert_eq!(envelope.operation, "geometry.query_entities");
    assert_eq!(envelope.op_version, "geometry.query_entities/v1");
    assert_eq!(envelope.data.entities.len(), 2);
    assert!(envelope.data.truncated);
    assert_eq!(envelope.data.entities[0].entity_id, 0);
    assert_eq!(envelope.data.entities[1].entity_id, 1);
}

#[test]
fn query_entities_region_not_found_maps_typed_error() {
    let asset = geometry_load("/assembly.step", SIMPLE_STEP.as_bytes()).expect("load should work");
    let error = geometry_query_entities_op(
        &asset,
        GeometryEntityQuery {
            region_id: Some("missing".to_string()),
            mesh_id: None,
            entity_kind: EntityKind::Node,
            limit: Some(8),
        },
        OperationContext::new(None, None),
    )
    .expect_err("query should fail");

    assert_eq!(error.error_code, "GEOMETRY_REGION_NOT_FOUND");
    assert_eq!(error.operation, "geometry.query_entities");
    assert_eq!(error.op_version, "geometry.query_entities/v1");
}

#[test]
fn capture_view_op_returns_typed_unsupported_error() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let error = geometry_capture_view_op(
        &asset,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 1024,
            height: 768,
        },
        OperationContext::new(Some("trace-g4".to_string()), None),
    )
    .expect_err("capture should be unsupported for now");

    assert_eq!(error.operation, "geometry.capture_view");
    assert_eq!(error.op_version, "geometry.capture_view/v1");
    assert_eq!(error.error_code, "GEOMETRY_CAPTURE_UNSUPPORTED");
    assert_eq!(error.trace_id.as_deref(), Some("trace-g4"));
}

#[test]
fn capture_view_op_validates_non_zero_dimensions() {
    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let error = geometry_capture_view_op(
        &asset,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 0,
            height: 768,
        },
        OperationContext::new(None, None),
    )
    .expect_err("zero width should be rejected");

    assert_eq!(error.error_code, "GEOMETRY_CAPTURE_INVALID_SPEC");
    assert_eq!(error.operation, "geometry.capture_view");
    assert_eq!(error.op_version, "geometry.capture_view/v1");
}

#[test]
fn capture_view_op_uses_installed_adapter() {
    struct TestCaptureAdapter;

    impl GeometryViewCaptureAdapter for TestCaptureAdapter {
        fn adapter_name(&self) -> &'static str {
            "test-capture"
        }

        fn capture(
            &self,
            _asset: &runmat_geometry_core::GeometryAsset,
            view_spec: &GeometryCaptureViewSpec,
        ) -> Result<GeometryCaptureViewResult, String> {
            Ok(GeometryCaptureViewResult {
                format: view_spec.format.clone(),
                width: view_spec.width,
                height: view_spec.height,
                payload: vec![1, 2, 3, 4],
            })
        }
    }

    static ADAPTER: TestCaptureAdapter = TestCaptureAdapter;
    let _guard = ThreadGeometryCaptureAdapterGuard::set(Some(&ADAPTER));

    let asset = geometry_load("/part.stl", TRIANGLE_STL.as_bytes()).expect("load should work");
    let envelope = geometry_capture_view_op(
        &asset,
        GeometryCaptureViewSpec {
            format: "png".to_string(),
            width: 640,
            height: 480,
        },
        OperationContext::new(Some("trace-g5".to_string()), None),
    )
    .expect("capture should succeed");

    assert_eq!(envelope.operation, "geometry.capture_view");
    assert_eq!(envelope.op_version, "geometry.capture_view/v1");
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-g5"));
    assert_eq!(envelope.data.format, "png");
    assert_eq!(envelope.data.width, 640);
    assert_eq!(envelope.data.height, 480);
    assert_eq!(envelope.data.payload, vec![1, 2, 3, 4]);
}
