use super::*;

#[test]
fn authoring_summary_exposes_region_readiness_without_raw_samples() {
    let mesh = minimal_evidence_mesh();
    let validation = AnalysisMeshValidationOptions {
        required_boundary_region_ids: vec!["fixed".to_string(), "loaded".to_string()],
        required_material_region_ids: vec!["solid".to_string(), "insert".to_string()],
        ..AnalysisMeshValidationOptions::default()
    };
    let evidence = build_mesh_evidence_artifact(&mesh, &validation);

    let summary = build_mesh_authoring_summary(&evidence);

    assert_eq!(
        summary.schema_version,
        MESH_AUTHORING_SUMMARY_SCHEMA_VERSION
    );
    assert_eq!(summary.mesh_id, mesh.mesh_id);
    assert!(!summary.solve_ready);
    assert!(summary.quality.meets_quality_thresholds);
    assert_eq!(summary.topology.volume_element_count, 1);
    assert_eq!(summary.recovery.boundary_face_recovery_ratio, 1.0);
    assert_eq!(
        summary.regions.required_boundary_region_ids,
        vec!["fixed".to_string(), "loaded".to_string()]
    );
    assert_eq!(
        summary.regions.missing_required_boundary_region_ids,
        vec!["loaded".to_string()]
    );
    assert_eq!(
        summary.regions.required_material_region_ids,
        vec!["insert".to_string(), "solid".to_string()]
    );
    assert_eq!(
        summary.regions.missing_required_material_region_ids,
        vec!["insert".to_string()]
    );
    assert_eq!(summary.regions.material_regions.len(), 1);
    assert_eq!(summary.regions.material_regions[0].region_id, "solid");
    assert!(summary.regions.material_regions[0].required);
    assert_eq!(summary.regions.boundary_regions.len(), 1);
    assert_eq!(summary.regions.boundary_regions[0].region_id, "fixed");
    assert!(summary.regions.boundary_regions[0].required);
    assert!(summary.regions.boundary_regions[0].fully_recovered);

    let encoded = serde_json::to_value(&summary).expect("serialize authoring summary");
    assert!(encoded.get("debug").is_none());
    assert!(encoded.get("sizing").is_none());
    assert!(encoded.get("mesh").is_none());
}

#[test]
fn authoring_summary_marks_failed_quality_thresholds() {
    let mesh = minimal_evidence_mesh();
    let validation = AnalysisMeshValidationOptions {
        quality: runmat_meshing_core::QualityThresholds {
            min_scaled_jacobian: 0.75,
            max_aspect_ratio: 1.5,
            ..runmat_meshing_core::QualityThresholds::default()
        },
        ..AnalysisMeshValidationOptions::default()
    };
    let evidence = build_mesh_evidence_artifact(&mesh, &validation);

    let summary = build_mesh_authoring_summary(&evidence);

    assert!(!summary.solve_ready);
    assert!(!summary.quality.meets_quality_thresholds);
    assert_eq!(
        summary.validation_error_code.as_deref(),
        Some("quality_threshold_failed")
    );
}
