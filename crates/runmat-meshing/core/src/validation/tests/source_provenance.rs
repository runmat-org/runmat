use super::*;
use fixtures::*;

#[test]
fn rejects_solid_mesh_without_boundary_source_face_provenance() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.boundary_faces[0].provenance.clear();

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("solid boundary faces must carry source-face provenance");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingBoundarySourceFaceProvenance {
            face_id: "f1".to_string(),
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_boundary_source_face_provenance"
    );
}

#[test]
fn accepts_solid_mesh_with_strict_boundary_source_edge_provenance() {
    let mesh = solid_tetrahedron_mesh_with_plc_input_evidence();

    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("strict source-edge provenance should pass when recovered edge evidence is present");
}

#[test]
fn rejects_solid_mesh_without_enough_boundary_source_edge_provenance() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.boundary_edges[0].provenance.clear();

    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("solid boundary edges must recover protected source-edge provenance");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingBoundarySourceEdgeProvenance {
            recovered_edge_count: 0,
            required_edge_count: 1,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_boundary_source_edge_provenance"
    );
}
