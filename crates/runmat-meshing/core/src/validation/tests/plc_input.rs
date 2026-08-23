use super::*;
use fixtures::*;

#[test]
fn rejects_solid_mesh_without_plc_input_evidence() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.backend = "solid".to_string();
    mesh.backend.algorithm = "plc_tetrahedron/v1".to_string();

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("solid Tetrahedron artifacts must prove PLC input evidence");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "missing_plc_nodes".to_string(),
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_plc_input_evidence"
    );
}

#[test]
fn accepts_solid_mesh_with_classified_plc_input_evidence() {
    let mesh = solid_tetrahedron_mesh_with_plc_input_evidence();

    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("classified PLC input evidence should satisfy solid validation");
}

#[test]
fn rejects_solid_mesh_without_plc_surface_boundary_node_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_surface_boundary_node_count = 0;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("solid PLC-fed meshes must expose consumed surface boundary nodes");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "missing_plc_surface_boundary_nodes".to_string(),
        }
    );
}

#[test]
fn rejects_solid_mesh_with_inconsistent_plc_surface_boundary_node_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_surface_boundary_node_count = mesh.backend.plc_input_node_count + 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("surface boundary node evidence must be bounded by PLC node count");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "inconsistent_plc_surface_boundary_nodes".to_string(),
        }
    );
}

#[test]
fn accepts_solid_mesh_with_nested_shell_plc_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_node_count = 8;
    mesh.backend.plc_input_facet_count = 8;
    mesh.backend.plc_input_boundary_component_count = 2;
    mesh.backend.plc_input_boundary_component_node_count = 8;
    mesh.backend.plc_input_max_boundary_component_node_count = 4;
    mesh.backend.plc_input_nested_shell_count = 1;
    mesh.backend.plc_input_max_shell_nesting_depth = 1;

    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("general CDT evidence should admit nested PLC shells without family selection");
}

#[test]
fn accepts_solid_mesh_with_consistent_plc_cad_curve_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_protected_edge_count = 3;
    mesh.backend.plc_input_cad_curve_boundary_source_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_segment_count = 3;
    mesh.backend.plc_input_cad_curve_imported_edge_count = 1;
    mesh.backend.plc_input_cad_curve_evaluator_edge_count = 1;
    mesh.backend.plc_input_cad_curve_evaluator_sample_count = 4;
    mesh.backend.plc_input_cad_curve_live_query_edge_count = 1;
    mesh.backend.plc_input_cad_curve_live_query_sample_count = 2;
    mesh.backend
        .plc_input_cad_curve_rejected_evaluator_sample_count = 1;
    mesh.backend.plc_input_cad_curve_curvature_sized_edge_count = 1;
    mesh.backend.plc_input_cad_curve_curvature_sample_count = 2;

    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("consistent CAD curve PLC input evidence should satisfy validation");
}

#[test]
fn rejects_plc_cad_curve_evidence_without_source_edge_count() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_cad_curve_boundary_segment_count = 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("CAD curve PLC evidence must identify source edges");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "missing_plc_cad_curve_boundary_source_edges".to_string(),
        }
    );
}

#[test]
fn rejects_plc_cad_curve_source_edges_exceeding_protected_edges() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_protected_edge_count = 1;
    mesh.backend.plc_input_cad_curve_boundary_source_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_segment_count = 2;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("CAD curve source edges must be bounded by protected PLC edges");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "inconsistent_plc_cad_curve_source_edge_count".to_string(),
        }
    );
}

#[test]
fn rejects_plc_cad_curve_boundary_segments_below_source_edges() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_protected_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_source_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_segment_count = 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("each CAD curve source edge must contribute a boundary segment");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "inconsistent_plc_cad_curve_boundary_segment_count".to_string(),
        }
    );
}

#[test]
fn rejects_plc_cad_curve_live_query_edges_without_evaluator_edges() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_protected_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_source_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_segment_count = 2;
    mesh.backend.plc_input_cad_curve_live_query_edge_count = 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("live CAD curve queries must be backed by evaluator edges");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "inconsistent_plc_cad_curve_live_query_edge_count".to_string(),
        }
    );
}

#[test]
fn rejects_plc_cad_curve_samples_without_imported_or_evaluator_edges() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_protected_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_source_edge_count = 2;
    mesh.backend.plc_input_cad_curve_boundary_segment_count = 2;
    mesh.backend.plc_input_cad_curve_evaluator_sample_count = 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("CAD curve samples must be backed by imported or evaluator edges");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "missing_plc_cad_curve_sample_edge_evidence".to_string(),
        }
    );
}

#[test]
fn rejects_solid_mesh_with_unclassified_plc_shell_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_shell_nesting_classified = false;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("unclassified PLC shell evidence should fail");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "unclassified_plc_shell_nesting".to_string(),
        }
    );
}

#[test]
fn rejects_solid_mesh_without_tetrahedron_material_region_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.tetrahedron_material_region_count = 0;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("solid Tetrahedron artifacts must prove material ownership evidence");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "missing_tetrahedron_material_region_evidence".to_string(),
        }
    );
}

#[test]
fn rejects_solid_mesh_with_unclassified_tetrahedron_material_ownership_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.tetrahedron_unclassified_material_element_count = 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("solid Tetrahedron artifacts must not carry unclassified ownership evidence");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "unclassified_tetrahedron_material_ownership".to_string(),
        }
    );
}

#[test]
fn rejects_single_material_plc_with_multiple_generated_material_regions() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.tetrahedron_material_region_count = 2;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("single-material PLC evidence should preserve one generated material region");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "inconsistent_single_material_region_ownership".to_string(),
        }
    );
}
