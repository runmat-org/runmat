use super::*;
use fixtures::*;

#[test]
fn rejects_missing_required_boundary_region() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_boundary_region_ids: vec!["loaded".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("missing boundary region should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredBoundaryRegion {
            region_id: "loaded".to_string()
        }
    );
}

#[test]
fn rejects_required_boundary_region_without_recovered_face() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.boundary_faces[0].adjacent_volume_element_ids.clear();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_boundary_region_ids: vec!["fixed".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("unrecovered boundary region should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredBoundaryRegionRecovery {
            region_id: "fixed".to_string()
        }
    );
}

#[test]
fn rejects_missing_required_material_region() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["rib".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("missing material region should fail");
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_required_material_region"
    );
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredMaterialRegion {
            region_id: "rib".to_string()
        }
    );
}

#[test]
fn rejects_required_material_region_without_positive_volume() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.nodes[3].coordinates_m = mesh.nodes[0].coordinates_m;
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["mat_region".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("zero-volume material region should fail");
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_required_material_region_coverage"
    );
    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingRequiredMaterialRegionCoverage {
            region_id: "mat_region".to_string()
        }
    );
}
