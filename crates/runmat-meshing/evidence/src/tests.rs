use super::*;
use runmat_meshing_core::{
    contracts::RefinementIndicatorMode,
    contracts::{
        artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisBoundaryEdge, AnalysisBoundaryFace,
        AnalysisMeshNode, AnalysisMeshProvenance, AnalysisVolumeElement, BoundaryElementKind,
        VolumeElementKind,
    },
    contracts::{AnalysisMeshArtifact, MeshBackendSummary},
    quality::{AnalysisMeshQualityReport, ElementQuality},
    size::field::{
        AnisotropicSizingSample, MeshSizingField, SizingSample, SizingSampleApplication,
        SizingSampleRejection,
    },
    validation::AnalysisMeshValidationOptions,
};
use runmat_meshing_size::adaptive::{
    AdaptiveConvergenceStatus, AdaptiveIterationSummary, RefinementIndicatorStatus,
    RefinementIndicatorSummary, RefinementMarker, SizingFieldUpdate,
};
use std::collections::BTreeMap;

#[test]
fn evidence_summarizes_mesh_without_raw_sizing_samples() {
    let mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "mesh_1".to_string(),
        nodes: vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ],
        volume_elements: vec![AnalysisVolumeElement {
            element_id: "tetrahedron_1".to_string(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: "solid".to_string(),
            provenance: Vec::new(),
        }],
        boundary_faces: vec![AnalysisBoundaryFace {
            face_id: "face_1".to_string(),
            kind: BoundaryElementKind::Tri3,
            node_ids: vec![1, 2, 3],
            adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }],
        boundary_edges: vec![
            boundary_edge("edge_1", [1, 2]),
            boundary_edge("edge_2", [2, 3]),
            boundary_edge("edge_3", [1, 3]),
        ],
        quality: AnalysisMeshQualityReport {
            min_scaled_jacobian: 0.5,
            min_exact_scaled_jacobian: 0.45,
            mean_aspect_ratio: 2.0,
            max_aspect_ratio: 2.0,
            inverted_element_count: 0,
            mean_boundary_projection_error_m: 0.0,
            max_boundary_projection_error_m: 0.0,
            elements: vec![ElementQuality {
                element_id: "tetrahedron_1".to_string(),
                scaled_jacobian: 0.5,
                exact_scaled_jacobian: 0.45,
                aspect_ratio: 2.0,
                volume_m3: 1.0 / 6.0,
            }],
        },
        sizing: MeshSizingField {
            growth_rate: Some(1.4),
            samples: vec![
                SizingSample {
                    position_m: [0.0, 0.0, 0.0],
                    target_size_m: 0.25,
                    reason: Some("load_region".to_string()),
                },
                SizingSample {
                    position_m: [0.5, 0.0, 0.0],
                    target_size_m: 0.2,
                    reason: Some("cad.curvature".to_string()),
                },
                SizingSample {
                    position_m: [0.0, 0.5, 0.0],
                    target_size_m: 0.15,
                    reason: Some("cad.interface".to_string()),
                },
            ],
            anisotropic_samples: vec![
                AnisotropicSizingSample {
                    position_m: [0.2, 0.2, 0.2],
                    target_sizes_m: [0.02, 0.04, 0.08],
                    directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    reason: Some("boundary_layer".to_string()),
                },
                AnisotropicSizingSample {
                    position_m: [0.3, 0.2, 0.2],
                    target_sizes_m: [0.02, -0.04, 0.08],
                    directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    reason: Some("cad.proximity".to_string()),
                },
            ],
            applied_samples: vec![
                SizingSampleApplication {
                    position_m: [0.0, 0.0, 0.0],
                    target_size_m: 0.25,
                    inserted_breakpoint_count: 2,
                    reason: Some("load_region".to_string()),
                    detail: Some("sample detail should not be copied".to_string()),
                },
                SizingSampleApplication {
                    position_m: [0.5, 0.0, 0.0],
                    target_size_m: 0.2,
                    inserted_breakpoint_count: 0,
                    reason: Some("cad.curvature".to_string()),
                    detail: Some("sample detail should not be copied".to_string()),
                },
            ],
            rejected_samples: vec![SizingSampleRejection {
                position_m: [0.1, 0.0, 0.0],
                target_size_m: 0.1,
                status: "outside_bounds".to_string(),
                reason: Some("adaptive".to_string()),
                detail: Some("rejection detail should not be copied".to_string()),
            }],
            ..MeshSizingField::default()
        },
        backend: MeshBackendSummary {
            backend: "solid".to_string(),
            cad_topology_source: "semantic_cad".to_string(),
            cad_evaluation_source: "imported_evaluator_samples".to_string(),
            cad_vertex_count: 4,
            cad_edge_count: 6,
            cad_face_count: 4,
            cad_shell_count: 1,
            cad_volume_count: 1,
            cad_imported_face_count: 3,
            cad_evaluation_evaluator_face_count: 2,
            cad_evaluation_live_query_face_count: 0,
            cad_evaluation_exact_query_face_count: 1,
            cad_evaluation_missing_exact_query_face_count: 1,
            cad_evaluation_missing_derivative_query_face_count: 2,
            cad_evaluation_missing_curvature_query_face_count: 1,
            cad_evaluation_point_supported_face_count: 2,
            cad_evaluation_projection_supported_face_count: 2,
            cad_evaluation_normal_supported_face_count: 2,
            cad_evaluation_derivative_supported_face_count: 2,
            cad_evaluation_curvature_supported_face_count: 1,
            cad_evaluation_sample_count: 8,
            cad_evaluation_rejected_sample_count: 9,
            cad_projection_query_count: 12,
            cad_derivative_query_count: 6,
            cad_curvature_query_count: 5,
            cad_uv_domain_face_count: 10,
            cad_uv_projection_out_of_bounds_count: 2,
            cad_max_projection_error_m: 2.0e-6,
            cad_max_normal_deviation: 1.0e-5,
            cad_max_curvature_estimate_1_per_m: 0.125,
            surface_cad_face_count: 3,
            surface_source_edge_loop_count: 2,
            surface_closed_edge_loop_count: 1,
            surface_conforming_source_edge_count: 5,
            surface_missing_source_edge_count: 1,
            surface_exact_cad_sample_node_count: 4,
            surface_rejected_exact_cad_sample_count: 5,
            surface_max_cad_projection_error_m: 3.0e-6,
            tetrahedron_element_count: 12,
            tetrahedron_recovered_component_ratio: 1.0,
            tetrahedron_unrecovered_component_count: 0,
            tetrahedron_volume_coverage_ratio: 0.99,
            tetrahedron_refinement_pass_count: 2,
            tetrahedron_refinement_point_count: 5,
            tetrahedron_requested_refinement_point_count: 5,
            tetrahedron_accepted_requested_refinement_location_count: 5,
            tetrahedron_accepted_requested_refinement_point_count: 3,
            tetrahedron_accepted_requested_refinement_surrogate_point_count: 2,
            tetrahedron_rejected_requested_refinement_point_count: 1,
            tetrahedron_requested_refinement_rejected_by_reason: BTreeMap::from([(
                "quality_or_recovery".to_string(),
                1,
            )]),
            tetrahedron_dropped_requested_refinement_point_count: 2,
            tetrahedron_requested_refinement_dropped_by_reason: BTreeMap::from([(
                "not_retained_after_repair".to_string(),
                2,
            )]),
            tetrahedron_optimization_pass_count: 1,
            tetrahedron_smoothed_point_count: 2,
            tetrahedron_sliver_count: 1,
            tetrahedron_sliver_removed_count: 2,
            tetrahedron_optimization_target_seed_count: 7,
            tetrahedron_optimization_skipped_target_seed_count: 4,
            tetrahedron_optimization_rejected_edit_count: 3,
            tetrahedron_optimization_initial_max_aspect_ratio: 6.0,
            tetrahedron_optimization_final_max_aspect_ratio: 4.0,
            tetrahedron_optimization_initial_min_exact_scaled_jacobian: 0.32,
            tetrahedron_optimization_final_min_exact_scaled_jacobian: 0.40,
            tetrahedron_untangling_pass_count: 2,
            tetrahedron_untangling_initial_near_singular_count: 6,
            tetrahedron_untangling_final_near_singular_count: 1,
            tetrahedron_untangling_relocated_seed_count: 3,
            tetrahedron_untangling_reconnected_edge_star_count: 4,
            tetrahedron_untangling_reconnected_boundary_adjacent_cavity_count: 5,
            tetrahedron_untangling_reconnected_node_adjacent_cavity_count: 11,
            tetrahedron_exact_quality_repair_pass_count: 1,
            tetrahedron_exact_quality_reconnected_cavity_count: 2,
            tetrahedron_exact_quality_reconnection_quality_gain_count: 1,
            tetrahedron_exact_quality_face_neighbor_reconnected_cavity_count: 6,
            tetrahedron_exact_quality_connected_reconnected_cavity_count: 7,
            tetrahedron_exact_quality_node_adjacent_reconnected_cavity_count: 12,
            tetrahedron_exact_quality_boundary_adjacent_reconnected_cavity_count: 8,
            tetrahedron_exact_quality_expanded_connected_reconnected_cavity_count: 9,
            tetrahedron_exact_quality_split_cavity_count: 3,
            tetrahedron_exact_quality_seed_star_collapse_count: 4,
            tetrahedron_exact_quality_seed_star_relocation_count: 5,
            tetrahedron_exact_quality_unrepaired_total_count: 9,
            tetrahedron_exact_quality_unrepaired_general_cavity_count: 1,
            tetrahedron_exact_quality_unrepaired_boundary_adjacent_count: 6,
            tetrahedron_exact_quality_unrepaired_node_adjacent_count: 10,
            tetrahedron_exact_quality_unrepaired_interior_seed_count: 7,
            tetrahedron_exact_quality_unrepaired_edge_star_count: 8,
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "test".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
        },
    };

    let validation = AnalysisMeshValidationOptions {
        max_volume_element_count: Some(7),
        max_volume_component_count: Some(1),
        coverage_sample_points_m: vec![[0.1, 0.1, 0.1]],
        min_coverage_sample_ratio: 1.0,
        require_no_unrecovered_tetrahedron_components: true,
        ..AnalysisMeshValidationOptions::default()
    };
    let evidence = build_mesh_evidence_artifact(&mesh, &validation);

    assert_eq!(evidence.schema_version, MESH_EVIDENCE_SCHEMA_VERSION);
    assert!(evidence.validation.solve_ready);
    assert_eq!(evidence.validation.validation_error_code, None);
    assert_eq!(evidence.validation.validation_error_message, None);
    assert_eq!(evidence.cad.topology_source, "semantic_cad");
    assert_eq!(evidence.cad.evaluation_source, "imported_evaluator_samples");
    assert_eq!(evidence.cad.imported_face_count, 3);
    assert_eq!(evidence.cad.exact_query_face_count, 1);
    assert_eq!(evidence.cad.missing_exact_query_face_count, 1);
    assert_eq!(evidence.cad.missing_derivative_query_face_count, 2);
    assert_eq!(evidence.cad.missing_curvature_query_face_count, 1);
    assert_eq!(evidence.cad.point_evaluation_supported_face_count, 2);
    assert_eq!(evidence.cad.projection_supported_face_count, 2);
    assert_eq!(evidence.cad.normal_supported_face_count, 2);
    assert_eq!(evidence.cad.derivative_supported_face_count, 2);
    assert_eq!(evidence.cad.curvature_supported_face_count, 1);
    assert_eq!(evidence.cad.evaluator_sample_count, 8);
    assert_eq!(evidence.cad.evaluator_rejected_sample_count, 9);
    assert_eq!(evidence.cad.projection_query_count, 12);
    assert_eq!(evidence.cad.derivative_query_count, 6);
    assert_eq!(evidence.cad.curvature_query_count, 5);
    assert_eq!(evidence.cad.uv_domain_face_count, 10);
    assert_eq!(evidence.cad.uv_projection_out_of_bounds_count, 2);
    assert_eq!(evidence.cad.max_projection_error_m, 2.0e-6);
    assert_eq!(evidence.cad.max_normal_deviation, 1.0e-5);
    assert_eq!(evidence.cad.max_curvature_estimate_1_per_m, 0.125);
    assert_eq!(evidence.cad.surface_source_edge_loop_count, 2);
    assert_eq!(evidence.cad.surface_closed_edge_loop_count, 1);
    assert_eq!(evidence.cad.surface_conforming_source_edge_count, 5);
    assert_eq!(evidence.cad.surface_missing_source_edge_count, 1);
    assert_eq!(evidence.cad.surface_max_projection_error_m, 3.0e-6);
    assert_eq!(evidence.cad.surface_exact_cad_sample_node_count, 4);
    assert_eq!(evidence.cad.surface_rejected_exact_cad_sample_count, 5);
    assert_eq!(evidence.topology.node_count, 4);
    assert_eq!(evidence.adaptive.iteration_count, 0);
    assert_eq!(evidence.adaptive.latest_convergence_status, None);
    assert_eq!(evidence.adaptive.marker_count, 0);
    assert_eq!(evidence.adaptive.sizing_update_sample_count, 0);
    assert_eq!(evidence.validation.volume_element_count, 1);
    assert_eq!(evidence.validation.max_volume_element_count, Some(7));
    assert_eq!(evidence.validation.volume_component_count, 1);
    assert_eq!(evidence.validation.volume_component_element_counts, vec![1]);
    assert_eq!(evidence.validation.max_volume_component_count, Some(1));
    assert_eq!(evidence.validation.coverage_sample_count, 1);
    assert_eq!(evidence.validation.covered_coverage_sample_count, 1);
    assert_eq!(evidence.validation.coverage_sample_ratio, Some(1.0));
    assert_eq!(evidence.validation.min_coverage_sample_ratio, 1.0);
    assert_eq!(
        evidence.validation.coverage_sample_points_m,
        vec![[0.1, 0.1, 0.1]]
    );
    assert!(
        evidence
            .validation
            .require_no_unrecovered_tetrahedron_components
    );
    assert!(!evidence.validation.require_no_unrepaired_exact_quality);
    assert_eq!(
        evidence.validation.unrecovered_tetrahedron_component_count,
        0
    );
    assert_eq!(evidence.validation.unrepaired_exact_quality_total_count, 9);
    assert_eq!(
        evidence
            .validation
            .unrepaired_exact_quality_general_cavity_count,
        1
    );
    assert_eq!(
        evidence
            .validation
            .unrepaired_exact_quality_boundary_adjacent_count,
        6
    );
    assert_eq!(
        evidence
            .validation
            .unrepaired_exact_quality_node_adjacent_count,
        10
    );
    assert_eq!(
        evidence
            .validation
            .unrepaired_exact_quality_interior_seed_count,
        7
    );
    assert_eq!(
        evidence.validation.unrepaired_exact_quality_edge_star_count,
        8
    );
    assert_eq!(evidence.sizing.inserted_breakpoint_count, 2);
    assert_eq!(
        evidence.sizing.requested_tetrahedron_refinement_point_count,
        5
    );
    assert_eq!(
        evidence
            .sizing
            .accepted_requested_tetrahedron_refinement_location_count,
        5
    );
    assert_eq!(
        evidence
            .sizing
            .accepted_requested_tetrahedron_refinement_point_count,
        3
    );
    assert_eq!(
        evidence
            .sizing
            .rejected_requested_tetrahedron_refinement_point_count,
        1
    );
    assert_eq!(
        evidence
            .sizing
            .requested_tetrahedron_refinement_rejected_by_reason
            .get("quality_or_recovery"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .sizing
            .dropped_requested_tetrahedron_refinement_point_count,
        2
    );
    assert_eq!(
        evidence
            .sizing
            .requested_tetrahedron_refinement_dropped_by_reason
            .get("not_retained_after_repair"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .sizing
            .accepted_requested_tetrahedron_refinement_surrogate_point_count,
        2
    );
    assert_eq!(
        evidence
            .sizing
            .accepted_requested_tetrahedron_refinement_exact_point_count,
        1
    );
    assert_eq!(
        evidence
            .sizing
            .requested_tetrahedron_refinement_acceptance_ratio,
        Some(0.6)
    );
    assert_eq!(
        evidence
            .sizing
            .requested_tetrahedron_refinement_rejection_ratio,
        Some(0.2)
    );
    assert_eq!(
        evidence
            .sizing
            .requested_tetrahedron_refinement_surrogate_ratio,
        Some(2.0 / 3.0)
    );
    assert_eq!(evidence.sizing.sample_count, 3);
    assert_eq!(evidence.sizing.generated_cad_sample_count, 2);
    assert_eq!(evidence.sizing.anisotropic_sample_count, 2);
    assert_eq!(evidence.sizing.valid_anisotropic_sample_count, 1);
    assert_eq!(evidence.sizing.invalid_anisotropic_sample_count, 1);
    assert_eq!(
        evidence.sizing.anisotropic_by_reason.get("boundary_layer"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .sizing
            .invalid_anisotropic_by_reason
            .get("cad.proximity"),
        Some(&1)
    );
    assert_eq!(
        evidence.sizing.generated_cad_by_reason.get("cad.curvature"),
        Some(&1)
    );
    assert_eq!(
        evidence.sizing.generated_cad_by_reason.get("cad.interface"),
        Some(&1)
    );
    assert_eq!(
        evidence.sizing.applied_by_reason.get("load_region"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .sizing
            .inserted_breakpoint_by_reason
            .get("load_region"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .sizing
            .uninserted_sample_by_reason
            .get("cad.curvature"),
        Some(&1)
    );
    assert_eq!(evidence.tetrahedron_recovery.element_count, 12);
    assert_eq!(evidence.tetrahedron_recovery.recovered_component_ratio, 1.0);
    assert_eq!(evidence.tetrahedron_recovery.volume_coverage_ratio, 0.99);
    assert_eq!(evidence.tetrahedron_recovery.refinement_pass_count, 2);
    assert_eq!(evidence.tetrahedron_recovery.refinement_point_count, 5);
    assert_eq!(evidence.tetrahedron_recovery.optimization_pass_count, 1);
    assert_eq!(evidence.tetrahedron_recovery.smoothed_point_count, 2);
    assert_eq!(evidence.tetrahedron_recovery.sliver_count, 1);
    assert_eq!(evidence.tetrahedron_recovery.sliver_removed_count, 2);
    assert_eq!(
        evidence.tetrahedron_recovery.optimization_target_seed_count,
        7
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .optimization_skipped_target_seed_count,
        4
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .optimization_rejected_edit_count,
        3
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .optimization_initial_max_aspect_ratio,
        6.0
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .optimization_final_max_aspect_ratio,
        4.0
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .optimization_initial_min_exact_scaled_jacobian,
        0.32
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .optimization_final_min_exact_scaled_jacobian,
        0.40
    );
    assert_eq!(evidence.tetrahedron_recovery.untangling_pass_count, 2);
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .untangling_initial_near_singular_count,
        6
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .untangling_final_near_singular_count,
        1
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .untangling_relocated_seed_count,
        3
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .untangling_reconnected_edge_star_count,
        4
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .untangling_reconnected_boundary_adjacent_cavity_count,
        5
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .untangling_reconnected_node_adjacent_cavity_count,
        11
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_repair_pass_count,
        1
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_reconnected_cavity_count,
        2
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_reconnection_quality_gain_count,
        1
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_face_neighbor_reconnected_cavity_count,
        6
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_connected_reconnected_cavity_count,
        7
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_node_adjacent_reconnected_cavity_count,
        12
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_boundary_adjacent_reconnected_cavity_count,
        8
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_expanded_connected_reconnected_cavity_count,
        9
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_split_cavity_count,
        3
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_seed_star_collapse_count,
        4
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_seed_star_relocation_count,
        5
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_unrepaired_total_count,
        9
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_unrepaired_general_cavity_count,
        1
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_unrepaired_boundary_adjacent_count,
        6
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_unrepaired_node_adjacent_count,
        10
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_unrepaired_interior_seed_count,
        7
    );
    assert_eq!(
        evidence
            .tetrahedron_recovery
            .exact_quality_unrepaired_edge_star_count,
        8
    );
    assert_eq!(evidence.sizing.growth_rate, Some(1.4));
    assert_eq!(
        evidence.sizing.rejected_by_status.get("outside_bounds"),
        Some(&1)
    );
    assert_eq!(
        evidence.regions.boundary_region_face_counts.get("fixed"),
        Some(&1)
    );
    assert_eq!(
        evidence.regions.material_region_volume_m3.get("solid"),
        Some(&(1.0 / 6.0))
    );
    assert_eq!(
        evidence
            .regions
            .boundary_region_recovered_face_counts
            .get("fixed"),
        Some(&1)
    );
    assert_eq!(evidence.quality.min_exact_scaled_jacobian, 0.45);
    assert_eq!(evidence.quality.scaled_jacobian_p05, Some(0.5));
    assert_eq!(evidence.quality.scaled_jacobian_p50, Some(0.5));
    assert_eq!(evidence.quality.scaled_jacobian_p95, Some(0.5));
    assert_eq!(evidence.quality.exact_scaled_jacobian_p05, Some(0.45));
    assert_eq!(evidence.quality.exact_scaled_jacobian_p50, Some(0.45));
    assert_eq!(evidence.quality.exact_scaled_jacobian_p95, Some(0.45));
    assert_eq!(evidence.quality.aspect_ratio_p50, Some(2.0));
    assert_eq!(evidence.quality.aspect_ratio_p95, Some(2.0));
    assert_eq!(
        evidence
            .quality
            .exact_scaled_jacobian_bins
            .get("0_35_to_0_65"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .validation
            .boundary_recovery
            .boundary_edge_recovery_ratio,
        1.0
    );

    let encoded = serde_json::to_value(&evidence).expect("serialize evidence");
    assert!(encoded.get("sizing").is_some());
    assert!(encoded.get("debug").is_none());
    assert_eq!(
        encoded["cad"]["evaluation_source"],
        serde_json::Value::String("imported_evaluator_samples".to_string())
    );
    assert!(
        encoded
            .to_string()
            .contains("sample detail should not be copied")
            == false
    );

    let failed_validation = AnalysisMeshValidationOptions {
        max_volume_element_count: Some(0),
        ..AnalysisMeshValidationOptions::default()
    };
    let failed_evidence = build_mesh_evidence_artifact(&mesh, &failed_validation);
    assert!(!failed_evidence.validation.solve_ready);
    assert_eq!(
        failed_evidence.validation.validation_error_code.as_deref(),
        Some("element_budget_exceeded")
    );
    assert!(failed_evidence
        .validation
        .validation_error_message
        .as_deref()
        .is_some_and(|message| message.contains("ElementBudgetExceeded")));

    let mut stale_validation = evidence.validation.clone();
    stale_validation.solve_ready = false;
    stale_validation.validation_error_code = Some("stale".to_string());
    stale_validation.validation_error_message = Some("stale".to_string());
    stale_validation.volume_element_count = 999;
    stale_validation.volume_component_count = 999;
    stale_validation
        .boundary_recovery
        .boundary_edge_recovery_ratio = 0.0;
    let refreshed_evidence =
        build_mesh_evidence_artifact_with_validation_evidence(&mesh, stale_validation);
    assert!(refreshed_evidence.validation.solve_ready);
    assert_eq!(refreshed_evidence.validation.validation_error_code, None);
    assert_eq!(refreshed_evidence.validation.volume_element_count, 1);
    assert_eq!(refreshed_evidence.validation.volume_component_count, 1);
    assert_eq!(
        refreshed_evidence
            .validation
            .boundary_recovery
            .boundary_edge_recovery_ratio,
        1.0
    );
}

#[test]
fn evidence_summarizes_adaptive_iterations_without_raw_marker_details() {
    let mut mesh = minimal_evidence_mesh();
    mesh.adaptive_iterations = vec![
        AdaptiveIterationSummary {
            iteration_index: 0,
            node_count: 4,
            element_count: 1,
            convergence_status: AdaptiveConvergenceStatus::Pending,
            indicators: vec![RefinementIndicatorSummary {
                namespace: "structural".to_string(),
                name: "load_regions".to_string(),
                requested_mode: RefinementIndicatorMode::Auto,
                status: RefinementIndicatorStatus::Used,
                detail: Some("field available".to_string()),
            }],
            markers: vec![RefinementMarker {
                entity_id: "face_1".to_string(),
                weight: 1.0,
                reason: "structural.load_regions".to_string(),
            }],
            sizing_update: SizingFieldUpdate {
                samples: vec![SizingSample {
                    position_m: [0.0, 0.0, 1.0],
                    target_size_m: 0.25,
                    reason: Some("structural.load_regions".to_string()),
                }],
                min_size_m: None,
                max_size_m: None,
            },
        },
        AdaptiveIterationSummary {
            iteration_index: 1,
            node_count: 5,
            element_count: 2,
            convergence_status: AdaptiveConvergenceStatus::Converged,
            indicators: vec![
                RefinementIndicatorSummary {
                    namespace: "structural".to_string(),
                    name: "stress_gradient".to_string(),
                    requested_mode: RefinementIndicatorMode::Auto,
                    status: RefinementIndicatorStatus::Used,
                    detail: None,
                },
                RefinementIndicatorSummary {
                    namespace: "thermal".to_string(),
                    name: "temperature_gradient".to_string(),
                    requested_mode: RefinementIndicatorMode::Auto,
                    status: RefinementIndicatorStatus::SkippedMissingField,
                    detail: Some("required recovered field is unavailable".to_string()),
                },
            ],
            markers: vec![
                RefinementMarker {
                    entity_id: "tetrahedron_1".to_string(),
                    weight: 1.0,
                    reason: "structural.stress_gradient".to_string(),
                },
                RefinementMarker {
                    entity_id: "tetrahedron_2".to_string(),
                    weight: 0.5,
                    reason: "structural.stress_gradient".to_string(),
                },
            ],
            sizing_update: SizingFieldUpdate {
                samples: vec![
                    SizingSample {
                        position_m: [0.2, 0.2, 0.2],
                        target_size_m: 0.2,
                        reason: Some("structural.stress_gradient".to_string()),
                    },
                    SizingSample {
                        position_m: [0.4, 0.2, 0.2],
                        target_size_m: 0.2,
                        reason: Some("structural.stress_gradient".to_string()),
                    },
                ],
                min_size_m: None,
                max_size_m: None,
            },
        },
    ];

    let evidence = build_mesh_evidence_artifact(&mesh, &AnalysisMeshValidationOptions::default());

    assert_eq!(evidence.topology.adaptive_iteration_count, 2);
    assert_eq!(evidence.adaptive.iteration_count, 2);
    assert_eq!(evidence.adaptive.latest_iteration_index, Some(1));
    assert_eq!(
        evidence.adaptive.latest_convergence_status.as_deref(),
        Some("converged")
    );
    assert_eq!(evidence.adaptive.latest_indicator_count, 2);
    assert_eq!(evidence.adaptive.latest_used_indicator_count, 1);
    assert_eq!(evidence.adaptive.latest_marker_count, 2);
    assert_eq!(evidence.adaptive.latest_sizing_update_sample_count, 2);
    assert_eq!(evidence.adaptive.marker_count, 3);
    assert_eq!(evidence.adaptive.sizing_update_sample_count, 3);
    assert_eq!(
        evidence.adaptive.latest_indicator_status_counts.get("used"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .adaptive
            .latest_indicator_status_counts
            .get("skipped_missing_field"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .adaptive
            .latest_marker_by_reason
            .get("structural.stress_gradient"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .adaptive
            .latest_sizing_update_by_reason
            .get("structural.stress_gradient"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .adaptive
            .marker_by_reason
            .get("structural.load_regions"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .adaptive
            .marker_by_reason
            .get("structural.stress_gradient"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .adaptive
            .sizing_update_by_reason
            .get("structural.load_regions"),
        Some(&1)
    );
}

#[cfg(feature = "dev-evidence")]
#[test]
fn dev_mesh_evidence_caps_debug_events() {
    let mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "debug_mesh".to_string(),
        nodes: vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ],
        volume_elements: vec![volume_element("tetrahedron_1", [1, 2, 3, 4])],
        boundary_faces: vec![boundary_face("face_1", [1, 2, 3])],
        boundary_edges: vec![
            boundary_edge("edge_1", [1, 2]),
            boundary_edge("edge_2", [2, 3]),
            boundary_edge("edge_3", [1, 3]),
        ],
        sizing: MeshSizingField::default(),
        quality: quality_report(),
        backend: MeshBackendSummary::default(),
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "test".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
        },
    };

    let evidence = build_mesh_evidence_artifact_with_debug(
        &mesh,
        &AnalysisMeshValidationOptions::default(),
        vec![
            MeshDebugEvent::new("surface", "info", "surface recovery accepted"),
            MeshDebugEvent::new("volume", "warning", "Tetrahedron quality improved"),
            MeshDebugEvent::new("validation", "info", "solve readiness checked"),
        ],
        2,
    );

    let debug = evidence.debug.expect("dev evidence should include debug");
    assert_eq!(debug.event_cap, 2);
    assert_eq!(debug.event_count, 3);
    assert_eq!(debug.emitted_event_count, 2);
    assert_eq!(debug.truncated_event_count, 1);
    assert_eq!(debug.events[0].stage, "surface");
    assert_eq!(debug.events[1].stage, "volume");

    let encoded = serde_json::to_value(&debug).expect("serialize debug evidence");
    assert_eq!(encoded["events"].as_array().map(Vec::len), Some(2));
}

fn node(node_id: u32, coordinates_m: [f64; 3]) -> AnalysisMeshNode {
    AnalysisMeshNode {
        node_id,
        coordinates_m,
        provenance: Vec::new(),
    }
}

fn boundary_edge(edge_id: &str, node_ids: [u32; 2]) -> AnalysisBoundaryEdge {
    AnalysisBoundaryEdge {
        edge_id: edge_id.to_string(),
        node_ids,
        adjacent_boundary_face_ids: vec!["face_1".to_string()],
        region_ids: vec!["fixed".to_string()],
        provenance: Vec::new(),
    }
}

fn minimal_evidence_mesh() -> AnalysisMeshArtifact {
    AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: "adaptive_mesh".to_string(),
        nodes: vec![
            node(1, [0.0, 0.0, 0.0]),
            node(2, [1.0, 0.0, 0.0]),
            node(3, [0.0, 1.0, 0.0]),
            node(4, [0.0, 0.0, 1.0]),
        ],
        volume_elements: vec![AnalysisVolumeElement {
            element_id: "tetrahedron_1".to_string(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: vec![1, 2, 3, 4],
            material_region_id: "solid".to_string(),
            provenance: Vec::new(),
        }],
        boundary_faces: vec![AnalysisBoundaryFace {
            face_id: "face_1".to_string(),
            kind: BoundaryElementKind::Tri3,
            node_ids: vec![1, 2, 3],
            adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }],
        boundary_edges: vec![
            boundary_edge("edge_1", [1, 2]),
            boundary_edge("edge_2", [2, 3]),
            boundary_edge("edge_3", [1, 3]),
        ],
        sizing: MeshSizingField::default(),
        quality: AnalysisMeshQualityReport {
            min_scaled_jacobian: 0.5,
            min_exact_scaled_jacobian: 0.45,
            mean_aspect_ratio: 2.0,
            max_aspect_ratio: 2.0,
            inverted_element_count: 0,
            mean_boundary_projection_error_m: 0.0,
            max_boundary_projection_error_m: 0.0,
            elements: vec![ElementQuality {
                element_id: "tetrahedron_1".to_string(),
                scaled_jacobian: 0.5,
                exact_scaled_jacobian: 0.45,
                aspect_ratio: 2.0,
                volume_m3: 1.0 / 6.0,
            }],
        },
        backend: MeshBackendSummary::default(),
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "test".to_string(),
            source_geometry_id: "geo".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
        },
    }
}

#[cfg(feature = "dev-evidence")]
fn boundary_face(face_id: &str, node_ids: [u32; 3]) -> AnalysisBoundaryFace {
    AnalysisBoundaryFace {
        face_id: face_id.to_string(),
        kind: BoundaryElementKind::Tri3,
        node_ids: node_ids.into(),
        adjacent_volume_element_ids: vec!["tetrahedron_1".to_string()],
        region_ids: vec!["fixed".to_string()],
        provenance: Vec::new(),
    }
}

#[cfg(feature = "dev-evidence")]
fn volume_element(element_id: &str, node_ids: [u32; 4]) -> AnalysisVolumeElement {
    AnalysisVolumeElement {
        element_id: element_id.to_string(),
        kind: VolumeElementKind::Tetrahedron4,
        node_ids: node_ids.into(),
        material_region_id: "solid".to_string(),
        provenance: Vec::new(),
    }
}

#[cfg(feature = "dev-evidence")]
fn quality_report() -> AnalysisMeshQualityReport {
    AnalysisMeshQualityReport {
        min_scaled_jacobian: 0.5,
        min_exact_scaled_jacobian: 0.45,
        mean_aspect_ratio: 2.0,
        max_aspect_ratio: 2.0,
        inverted_element_count: 0,
        mean_boundary_projection_error_m: 0.0,
        max_boundary_projection_error_m: 0.0,
        elements: vec![ElementQuality {
            element_id: "tetrahedron_1".to_string(),
            scaled_jacobian: 0.5,
            exact_scaled_jacobian: 0.45,
            aspect_ratio: 2.0,
            volume_m3: 1.0 / 6.0,
        }],
    }
}
