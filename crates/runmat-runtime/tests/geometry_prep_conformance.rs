use runmat_runtime::geometry::{
    geometry_load_op, geometry_prep_for_analysis_op, GeometryPrepForAnalysisSpec,
    GeometryPrepProfile,
};
use runmat_runtime::operations::OperationContext;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

#[test]
fn geometry_prep_for_analysis_is_deterministic_for_fixture_inputs() {
    let spec = GeometryPrepForAnalysisSpec {
        profile: GeometryPrepProfile::AnalysisReady,
        target_element_budget: 120_000,
    };

    for (path, bytes) in [
        ("/fixtures/tri.stl", TRIANGLE_STL.as_bytes()),
        ("/fixtures/assy.step", SIMPLE_STEP.as_bytes()),
    ] {
        let geometry = geometry_load_op(path, bytes, OperationContext::new(None, None))
            .expect("geometry load should succeed");
        let first = geometry_prep_for_analysis_op(
            &geometry.data,
            spec.clone(),
            OperationContext::new(Some("trace-prep-1".to_string()), None),
        )
        .expect("first prep should succeed");
        let second = geometry_prep_for_analysis_op(
            &geometry.data,
            spec.clone(),
            OperationContext::new(Some("trace-prep-2".to_string()), None),
        )
        .expect("second prep should succeed");

        assert_ne!(first.data.prep_artifact_id, second.data.prep_artifact_id);
        assert_eq!(first.data.prep, second.data.prep);
        assert!(!first.data.prep_artifact_id.is_empty());
        assert_eq!(
            first.data.prep.schema_version,
            "geometry-prep-for-analysis/v1"
        );
        assert!(first.data.prep.quality.min_scaled_jacobian >= 0.5);
        assert_eq!(first.data.prep.quality.inverted_element_count, 0);
    }
}

#[test]
fn geometry_prep_for_analysis_preserves_region_mapping_stability() {
    let geometry = geometry_load_op(
        "/fixtures/assy.step",
        SIMPLE_STEP.as_bytes(),
        OperationContext::new(None, None),
    )
    .expect("geometry load should succeed");
    let prep = geometry_prep_for_analysis_op(
        &geometry.data,
        GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(None, None),
    )
    .expect("prep should succeed");

    assert!(!prep.data.prep.region_mappings.is_empty());
    for mapping in &prep.data.prep.region_mappings {
        assert!(!mapping.region_id.is_empty());
        assert!(!mapping.source_mesh_ids.is_empty());
        assert!(!mapping.prepared_mesh_ids.is_empty());
    }
}
