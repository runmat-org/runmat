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
fn authoring_summary_exposes_tetrahedron_generation_selection_counts() {
    let mut mesh = minimal_evidence_mesh();
    mesh.backend.tetrahedron_generation_family = "star_shaped_polyhedron".to_string();
    mesh.backend.tetrahedron_generation_attempted_family_count = 5;
    mesh.backend.tetrahedron_generation_rejected_family_count = 4;
    mesh.backend.tetrahedron_generation_selected_family_index = 5;
    mesh.backend
        .tetrahedron_generation_interior_support_candidate_count = 29;
    mesh.backend
        .tetrahedron_generation_interior_support_accepted_count = 1;
    let evidence = build_mesh_evidence_artifact(&mesh, &AnalysisMeshValidationOptions::default());

    let summary = build_mesh_authoring_summary(&evidence);

    assert_eq!(
        summary.tetrahedron_generation_family,
        "star_shaped_polyhedron"
    );
    assert_eq!(summary.tetrahedron_generation_attempted_family_count, 5);
    assert_eq!(summary.tetrahedron_generation_rejected_family_count, 4);
    assert_eq!(summary.tetrahedron_generation_selected_family_index, 5);
    assert_eq!(
        summary.tetrahedron_generation_interior_support_candidate_count,
        29
    );
    assert_eq!(
        summary.tetrahedron_generation_interior_support_accepted_count,
        1
    );

    let encoded = serde_json::to_value(&summary).expect("serialize authoring summary");
    assert_eq!(
        encoded["tetrahedron_generation_attempted_family_count"].as_u64(),
        Some(5)
    );
    assert_eq!(
        encoded["tetrahedron_generation_rejected_family_count"].as_u64(),
        Some(4)
    );
    assert_eq!(
        encoded["tetrahedron_generation_selected_family_index"].as_u64(),
        Some(5)
    );
    assert_eq!(
        encoded["tetrahedron_generation_interior_support_candidate_count"].as_u64(),
        Some(29)
    );
    assert_eq!(
        encoded["tetrahedron_generation_interior_support_accepted_count"].as_u64(),
        Some(1)
    );
    assert!(encoded.get("debug").is_none());
    assert!(encoded.get("mesh").is_none());
}

#[test]
fn authoring_summary_exposes_nested_tetrahedron_shell_counts() {
    let mut mesh = minimal_evidence_mesh();
    mesh.backend.tetrahedron_generation_family = "nested_tetrahedron_shell".to_string();
    mesh.backend
        .tetrahedron_generation_nested_shell_outer_node_count = 4;
    mesh.backend
        .tetrahedron_generation_nested_shell_inner_node_count = 4;
    mesh.backend
        .tetrahedron_generation_nested_shell_generated_node_count = 1;
    mesh.backend
        .tetrahedron_generation_nested_shell_refill_boundary_face_count = 8;
    mesh.backend
        .tetrahedron_generation_nested_shell_boundary_centroid_refinement_attempt_count = 1;
    mesh.backend
        .tetrahedron_generation_nested_shell_boundary_centroid_refinement_rejected_count = 1;
    mesh.backend
        .tetrahedron_generation_nested_shell_boundary_exact_cover_refill_count = 1;
    mesh.backend
        .tetrahedron_generation_nested_shell_boundary_centroid_refinement_refill_count = 0;
    mesh.backend
        .tetrahedron_generation_nested_shell_barycentric_partition_refill_count = 0;
    mesh.backend
        .tetrahedron_generation_nested_shell_outer_facet_count = 4;
    mesh.backend
        .tetrahedron_generation_nested_shell_inner_facet_count = 4;
    let evidence = build_mesh_evidence_artifact(&mesh, &AnalysisMeshValidationOptions::default());

    let summary = build_mesh_authoring_summary(&evidence);

    assert_eq!(summary.nested_tetrahedron_shell.outer_node_count, 4);
    assert_eq!(summary.nested_tetrahedron_shell.inner_node_count, 4);
    assert_eq!(summary.nested_tetrahedron_shell.generated_node_count, 1);
    assert_eq!(
        summary
            .nested_tetrahedron_shell
            .boundary_exact_cover_refill_count,
        1
    );
    assert_eq!(summary.nested_tetrahedron_shell.outer_facet_count, 4);
    assert_eq!(summary.nested_tetrahedron_shell.inner_facet_count, 4);

    let encoded = serde_json::to_value(&summary).expect("serialize authoring summary");
    assert_eq!(
        encoded["nested_tetrahedron_shell"]["outer_node_count"].as_u64(),
        Some(4)
    );
    assert_eq!(
        encoded["nested_tetrahedron_shell"]["boundary_exact_cover_refill_count"].as_u64(),
        Some(1)
    );
    assert!(encoded.get("debug").is_none());
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
