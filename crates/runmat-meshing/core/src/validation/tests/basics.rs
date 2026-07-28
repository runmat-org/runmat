use super::*;
use fixtures::*;

#[test]
fn accepts_minimal_valid_tetrahedron4_mesh() {
    let mesh = valid_tetrahedron_mesh();
    validate_analysis_mesh(&mesh, QualityThresholds::default()).expect("mesh should validate");
}

#[test]
fn validation_options_round_trip_with_required_regions() {
    let options = AnalysisMeshValidationOptions {
        required_boundary_region_ids: vec!["fixed".to_string(), "loaded".to_string()],
        required_material_region_ids: vec!["solid".to_string()],
        max_volume_element_count: Some(42),
        coverage_sample_points_m: vec![[0.25, 0.25, 0.25]],
        min_coverage_sample_ratio: 0.75,
        require_no_unrecovered_tetrahedron_components: true,
        ..AnalysisMeshValidationOptions::default()
    };

    let encoded = serde_json::to_value(&options).expect("validation options should serialize");
    let decoded: AnalysisMeshValidationOptions =
        serde_json::from_value(encoded).expect("validation options should deserialize");

    assert_eq!(decoded, options);
}

#[test]
fn rejects_empty_volume_elements() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.volume_elements.clear();
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("empty volume elements should fail");
    assert_eq!(err, AnalysisMeshValidationError::EmptyVolumeElements);
}

#[test]
fn rejects_mesh_that_exceeds_element_budget() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            max_volume_element_count: Some(0),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("element budget overrun should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::ElementBudgetExceeded {
            element_count: 1,
            max_element_count: 0,
        }
    );
}
