#![allow(dead_code)]

use super::*;
use crate::{
    field_mapping::{
        map_nodal_vector_field_to_boundary_faces, map_nodal_vector_field_to_boundary_nodes,
        map_volume_scalar_field_to_boundary_faces,
    },
    validation::volume_component_count,
};
use runmat_geometry_core::{
    CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
    CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, GeometryAsset,
    GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping, SourceGeometry,
    SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
};

#[test]
fn preparation_runs_topology_curve_surface_plc_and_tet_stages() {
    let preparation = prepare_solid_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
        .expect("solid preparation should run");

    assert_eq!(preparation.topology.faces.len(), 12);
    assert_eq!(preparation.cad_topology.report.vertex_count, 8);
    assert_eq!(preparation.cad_topology.report.edge_count, 18);
    assert_eq!(preparation.cad_topology.report.face_count, 12);
    assert_eq!(preparation.cad_topology.report.closed_shell_count, 1);
    assert_eq!(preparation.cad_evaluation.report.face_frame_count, 12);
    assert_eq!(
        preparation
            .cad_evaluation_report
            .missing_exact_query_face_count,
        0
    );
    assert_eq!(preparation.cad_evaluation_report.projection_query_count, 36);
    assert_eq!(
        preparation.cad_evaluation_report.max_projection_error_m,
        0.0
    );
    assert_eq!(preparation.surface.elements.len(), 768);
    assert_eq!(preparation.surface.exact_cad_sample_node_count, 0);
    assert_eq!(preparation.surface.rejected_exact_cad_sample_count, 0);
    assert_eq!(surface_cad_face_count(&preparation.surface), 12);
    assert_eq!(
        surface_max_cad_projection_error_m(&preparation.surface),
        0.0
    );
    assert!(preparation
        .surface
        .elements
        .iter()
        .all(|element| element.cad_face_id.is_some()));
    assert_eq!(preparation.surface_validation.source_edge_loop_count, 1);
    assert_eq!(
        preparation.surface_validation.closed_source_edge_loop_count,
        1
    );
    assert_eq!(preparation.surface_validation.face_coverage_ratio, 1.0);
    assert_eq!(preparation.surface_recovery.surface_element_count, 768);
    assert_eq!(preparation.surface_recovery.open_edge_count, 0);
    assert_eq!(preparation.surface_recovery.nonmanifold_edge_count, 0);
    assert_eq!(preparation.surface_recovery.source_face_coverage_ratio, 1.0);
    assert!(preparation
        .protected_boundary_complex
        .validation
        .valid_for_volume_meshing());
    assert_eq!(preparation.protected_boundary_complex.facets.len(), 768);
    assert_eq!(
        preparation
            .protected_boundary_complex
            .evidence
            .entity_counts["facets"],
        768
    );
    assert!(!preparation
        .protected_boundary_complex
        .protected_edges
        .is_empty());
    assert_eq!(
        preparation.initial_tet_mesh.elements.len(),
        preparation.protected_boundary_complex.facets.len()
    );
    assert_eq!(
        preparation.initial_tet_mesh.nodes.len(),
        preparation.protected_boundary_complex.nodes.len() + 1
    );
    assert!(!preparation.initial_tet_mesh.recovery_complete);
    assert!(!preparation.initial_tet_mesh.quality_optimized);
    assert_eq!(
        preparation.recovery_queue.evidence.entity_counts["source_face_items"],
        preparation.protected_boundary_complex.facets.len()
    );
    assert_eq!(
        preparation.recovery_queue.evidence.entity_counts["source_edge_items"],
        preparation.protected_boundary_complex.protected_edges.len()
    );
    assert!(preparation.recovery_queue.evidence.entity_counts["material_interface_items"] > 0);
    assert_eq!(preparation.tet_stage.volume_component_count, 1);
    assert_eq!(preparation.solver_tet_mesh.elements.len(), 6);
    assert_eq!(preparation.tet_stage.interior_seed_point_count, 8);
    assert_eq!(preparation.tet_stage.recovered_component_ratio, 1.0);
    assert!((preparation.tet_stage.expected_volume_m3 - 1.0).abs() < 1.0e-12);
    assert!((preparation.tet_stage.expected_boundary_area_m2 - 6.0).abs() < 1.0e-12);
    assert!(!preparation.curves.elements.is_empty());
}

#[test]
fn solid_mesh_generates_analysis_mesh_artifact_from_solver_tet_mesh() {
    let preparation = prepare_solid_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
        .expect("solid preparation should generate");
    let mesh = generate_solid_analysis_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
        .expect("solid mesh should generate from native solver TetMesh");

    crate::validate_analysis_mesh(&mesh, crate::QualityThresholds::default())
        .expect("solid topology-first mesh should validate");
    assert!(mesh.nodes.len() > 9);
    assert_eq!(
        mesh.volume_elements.len(),
        preparation.solver_tet_mesh.elements.len()
    );
    assert_eq!(mesh.volume_elements.len(), 6);
    assert_eq!(mesh.boundary_faces.len(), 768);
    assert_eq!(mesh.boundary_edges.len(), 1152);
    assert!(mesh
        .boundary_edges
        .iter()
        .all(|edge| !edge.adjacent_boundary_face_ids.is_empty()));
    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.backend.algorithm, "plc_tet/v1");
    assert_eq!(mesh.backend.source_topology_face_count, 12);
    assert_eq!(mesh.backend.cad_topology_source, "generic_cad_mesh");
    assert_eq!(mesh.backend.cad_vertex_count, 8);
    assert_eq!(mesh.backend.cad_edge_count, 18);
    assert_eq!(mesh.backend.cad_face_count, 12);
    assert_eq!(mesh.backend.cad_closed_shell_count, 1);
    assert_eq!(mesh.backend.cad_imported_face_count, 0);
    assert_eq!(mesh.backend.cad_evaluator_face_count, 0);
    assert_eq!(
        mesh.backend.cad_evaluation_source,
        "planar_facet_approximation"
    );
    assert_eq!(mesh.backend.cad_face_frame_count, 12);
    assert_eq!(mesh.backend.cad_evaluation_evaluator_face_count, 0);
    assert_eq!(
        mesh.backend.cad_evaluation_missing_exact_query_face_count,
        0
    );
    assert_eq!(mesh.backend.cad_projection_query_count, 36);
    assert_eq!(mesh.backend.cad_max_projection_error_m, 0.0);
    assert_eq!(mesh.backend.surface_element_count, 768);
    assert_eq!(mesh.backend.surface_source_edge_loop_count, 1);
    assert_eq!(mesh.backend.surface_closed_edge_loop_count, 1);
    assert_eq!(mesh.backend.surface_conforming_source_edge_count, 18);
    assert_eq!(mesh.backend.surface_missing_source_edge_count, 0);
    assert_eq!(mesh.backend.surface_face_coverage_ratio, 1.0);
    assert_eq!(mesh.backend.surface_cad_face_count, 12);
    assert_eq!(mesh.backend.surface_exact_cad_sample_node_count, 0);
    assert_eq!(mesh.backend.surface_rejected_exact_cad_sample_count, 0);
    assert_eq!(mesh.backend.surface_max_cad_projection_error_m, 0.0);
    assert_eq!(mesh.backend.volume_component_count, 1);
    assert_eq!(mesh.backend.tet_element_count, mesh.volume_elements.len());
    assert_eq!(mesh.backend.tet_fan_fallback_component_count, 0);
    assert_eq!(mesh.backend.tet_recovered_component_ratio, 1.0);
    assert!((mesh.backend.tet_volume_coverage_ratio - 1.0).abs() < 1.0e-12);
    let validation_options =
        solid_validation_options(&preparation, &VolumeMeshingOptions::default());
    assert!(!validation_options.coverage_sample_points_m.is_empty());
    assert_eq!(validation_options.min_coverage_sample_ratio, 1.0);
    assert!(validation_options.require_no_unrepaired_exact_quality);
    assert!(validation_options
        .coverage_sample_points_m
        .iter()
        .all(|point| point.iter().all(|value| value.is_finite())));
    assert!(mesh.backend.tet_max_radius_edge_ratio.is_finite());
    assert!(mesh.backend.tet_min_exact_scaled_jacobian.is_finite());
    assert!(
        (mesh.backend.tet_min_exact_scaled_jacobian - mesh.quality.min_exact_scaled_jacobian).abs()
            < 1.0e-12
    );
    assert_eq!(
        mesh.backend.tet_exact_scaled_jacobian_below_threshold_count,
        mesh.quality
            .elements
            .iter()
            .filter(|element| element.exact_scaled_jacobian
                < QualityThresholds::default().min_scaled_jacobian)
            .count()
    );
    assert_eq!(
        mesh.backend.tet_exact_scaled_jacobian_below_threshold_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tet_exact_scaled_jacobian_bins
            .values()
            .sum::<usize>(),
        mesh.backend.tet_element_count
    );
    assert!(!mesh.backend.tet_exact_scaled_jacobian_bins.is_empty());
    assert!(mesh.backend.tet_optimization_pass_count <= 2);
    assert!(mesh.backend.tet_smoothed_point_count <= mesh.backend.interior_seed_point_count * 2);
    assert!(mesh
        .backend
        .tet_optimization_initial_max_aspect_ratio
        .is_finite());
    assert!(mesh
        .backend
        .tet_optimization_final_max_aspect_ratio
        .is_finite());
    assert!(
        mesh.backend.tet_optimization_final_max_aspect_ratio
            <= mesh.backend.tet_optimization_initial_max_aspect_ratio + 1.0e-12
    );
    assert!(mesh
        .backend
        .tet_optimization_initial_min_exact_scaled_jacobian
        .is_finite());
    assert!(
        mesh.backend
            .tet_optimization_final_min_exact_scaled_jacobian
            + 1.0e-12
            >= mesh
                .backend
                .tet_optimization_initial_min_exact_scaled_jacobian
    );
    assert!(mesh.quality.min_exact_scaled_jacobian.is_finite());
    assert!(mesh
        .quality
        .elements
        .iter()
        .all(|element| element.exact_scaled_jacobian.is_finite()));
    assert_eq!(
        mesh.quality.max_boundary_projection_error_m,
        mesh.backend.surface_max_cad_projection_error_m
    );
    assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
    assert_eq!(mesh.backend.boundary_edge_recovery_ratio, 1.0);
    assert_eq!(mesh.provenance.algorithm, "plc_tet/v1");
}

#[test]
fn solid_mesh_evidence_reports_native_recovery_and_optimization_summary() {
    let options = VolumeMeshingOptions::default();
    let preparation =
        prepare_solid_mesh(&cube_geometry(), &options).expect("solid preparation should generate");
    let mesh = analysis_mesh_from_preparation(&preparation, &options, None)
        .expect("solid mesh should generate with native summary evidence");
    let evidence = crate::build_mesh_evidence_artifact(
        &mesh,
        &solid_validation_options(&preparation, &options),
    );

    assert_eq!(mesh.backend.tet_untangling_pass_count, 0);
    assert_eq!(mesh.backend.tet_exact_quality_repair_pass_count, 0);
    assert_eq!(mesh.backend.tet_exact_quality_unrepaired_total_count, 0);
    assert_eq!(evidence.tet_recovery.untangling_pass_count, 0);
    assert_eq!(evidence.tet_recovery.exact_quality_repair_pass_count, 0);
    assert_eq!(
        evidence.tet_recovery.exact_quality_unrepaired_total_count,
        0
    );
}

#[test]
fn solid_mesh_preserves_boundary_and_material_region_provenance() {
    let options = VolumeMeshingOptions::default();
    let preparation =
        prepare_solid_mesh(&cube_geometry(), &options).expect("solid preparation should generate");
    let mesh = analysis_mesh_from_preparation(&preparation, &options, None)
        .expect("solid mesh should generate from multi-region fixture");
    let evidence = crate::build_mesh_evidence_artifact(
        &mesh,
        &solid_validation_options(&preparation, &options),
    );

    assert!(mesh
        .boundary_faces
        .iter()
        .any(|face| face.region_ids.iter().any(|region| region == "root")));
    assert!(mesh
        .boundary_faces
        .iter()
        .any(|face| face.region_ids.iter().any(|region| region == "tip")));
    assert!(mesh
        .boundary_edges
        .iter()
        .any(|edge| edge.region_ids.iter().any(|region| region == "root")));
    assert!(mesh
        .boundary_edges
        .iter()
        .any(|edge| edge.region_ids.iter().any(|region| region == "tip")));
    assert!(mesh
        .volume_elements
        .iter()
        .any(|element| element.material_region_id == "root"));
    assert!(mesh
        .volume_elements
        .iter()
        .any(|element| element.material_region_id == "tip"));
    assert!(mesh.boundary_faces.iter().all(|face| {
        face.provenance.iter().any(|provenance| {
            provenance.source_entity_kind == SourceEntityKind::Face
                && provenance.region_ids == face.region_ids
        })
    }));
    assert!(mesh.boundary_edges.iter().all(|edge| {
        edge.provenance.iter().any(|provenance| {
            provenance.source_entity_kind == SourceEntityKind::Edge
                && provenance.region_ids == edge.region_ids
        })
    }));
    assert!(mesh.volume_elements.iter().all(|element| {
        element.provenance.iter().any(|provenance| {
            provenance.source_entity_kind == SourceEntityKind::Face
                && provenance
                    .region_ids
                    .iter()
                    .any(|region| region == &element.material_region_id)
        })
    }));
    assert!(
        evidence
            .regions
            .boundary_region_face_counts
            .get("root")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .boundary_region_face_counts
            .get("tip")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .boundary_region_edge_counts
            .get("root")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .boundary_region_edge_counts
            .get("tip")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .material_region_element_counts
            .get("root")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .material_region_element_counts
            .get("tip")
            .copied()
            .unwrap_or_default()
            > 0
    );
}

#[test]
fn solid_mesh_preserves_regions_through_requested_refinement() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.75);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let sizing = MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [0.25, 0.25, 0.25],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [0.75, 0.75, 0.75],
                target_size_m: 0.25,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let mesh = generate_solid_analysis_mesh_with_sizing(&cube_geometry(), &options, &sizing)
        .expect("solid mesh should generate with requested refinement");
    let validation = AnalysisMeshValidationOptions {
        required_boundary_region_ids: vec!["root".to_string(), "tip".to_string()],
        required_material_region_ids: vec!["root".to_string(), "tip".to_string()],
        min_boundary_face_recovery_ratio: 1.0,
        min_boundary_edge_recovery_ratio: 1.0,
        require_no_fan_fallback: true,
        require_no_unrepaired_exact_quality: true,
        ..AnalysisMeshValidationOptions::default()
    };
    crate::validate_analysis_mesh_with_options(&mesh, validation.clone())
        .expect("required regions should remain selectable after requested refinement");
    let evidence = crate::build_mesh_evidence_artifact(&mesh, &validation);

    assert_eq!(mesh.backend.tet_requested_refinement_point_count, 2);
    assert_eq!(
        mesh.backend.tet_accepted_requested_refinement_point_count,
        0
    );
    assert_eq!(
        mesh.backend.tet_rejected_requested_refinement_point_count,
        2
    );
    for sample in &sizing.samples {
        let reason = sample.reason.as_deref().expect("reasoned sizing sample");
        assert!(
            mesh.sizing.rejected_samples.iter().any(|rejected| {
                rejected.reason.as_deref() == Some(reason)
                    && rejected.status == "not_inserted_by_tet_generation"
            }),
            "{reason} should be tracked as requested but not yet inserted"
        );
    }
    assert_eq!(
        mesh.backend
            .tet_requested_refinement_rejected_by_reason
            .get("native_tet_generator_has_no_requested_point_insertion"),
        Some(&2)
    );
    assert_eq!(
        evidence.sizing.requested_tet_refinement_point_count,
        mesh.backend.tet_requested_refinement_point_count
    );
    assert_eq!(
        evidence
            .sizing
            .accepted_requested_tet_refinement_point_count,
        mesh.backend.tet_accepted_requested_refinement_point_count
    );
    assert!(evidence.sizing.applied_by_reason.is_empty());
    assert_eq!(
        evidence
            .sizing
            .requested_tet_refinement_rejected_by_reason
            .get("native_tet_generator_has_no_requested_point_insertion"),
        Some(&2)
    );
    assert!(
        evidence
            .regions
            .boundary_region_recovered_face_counts
            .get("root")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .boundary_region_recovered_face_counts
            .get("tip")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .material_region_element_counts
            .get("root")
            .copied()
            .unwrap_or_default()
            > 0
    );
    assert!(
        evidence
            .regions
            .material_region_element_counts
            .get("tip")
            .copied()
            .unwrap_or_default()
            > 0
    );
}

#[test]
fn solid_boundary_patch_refinement_tracks_existing_tet_nodes() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(1.0);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let sizing = MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [1.0, 0.5, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [0.5, 0.0, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let mesh = generate_solid_analysis_mesh_with_sizing(&cube_geometry(), &options, &sizing)
        .expect("solid mesh should generate with boundary patch sizing");
    let validation = AnalysisMeshValidationOptions {
        required_boundary_region_ids: vec!["root".to_string(), "tip".to_string()],
        required_material_region_ids: vec!["root".to_string(), "tip".to_string()],
        min_boundary_face_recovery_ratio: 1.0,
        min_boundary_edge_recovery_ratio: 1.0,
        require_no_fan_fallback: true,
        require_no_unrepaired_exact_quality: true,
        ..AnalysisMeshValidationOptions::default()
    };
    crate::validate_analysis_mesh_with_options(&mesh, validation.clone())
        .expect("boundary patch refinement should preserve required regions");
    let evidence = crate::build_mesh_evidence_artifact(&mesh, &validation);

    assert_eq!(mesh.backend.tet_requested_refinement_point_count, 2);
    assert_eq!(
        mesh.backend.tet_accepted_requested_refinement_point_count,
        2
    );
    assert_eq!(
        mesh.backend
            .tet_accepted_requested_refinement_surrogate_point_count,
        0
    );
    for sample in &sizing.samples {
        let reason = sample.reason.as_deref().expect("reasoned boundary sample");
        assert!(
            mesh.sizing.applied_samples.iter().any(|applied| {
                applied.reason.as_deref() == Some(reason) && applied.inserted_breakpoint_count > 0
            }),
            "{reason} should be tracked on an existing Tet node"
        );
    }
    assert!(mesh
        .backend
        .tet_requested_refinement_rejected_by_reason
        .is_empty());
    assert!(mesh
        .backend
        .tet_requested_refinement_dropped_by_reason
        .is_empty());
    assert_eq!(
        evidence
            .sizing
            .accepted_requested_tet_refinement_surrogate_point_count,
        0
    );
    assert!(evidence
        .sizing
        .requested_tet_refinement_rejected_by_reason
        .is_empty());
    assert!(evidence
        .sizing
        .requested_tet_refinement_dropped_by_reason
        .is_empty());
    assert_eq!(
        evidence
            .sizing
            .applied_by_reason
            .get("structural.load_regions"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .sizing
            .applied_by_reason
            .get("structural.constraint_regions"),
        Some(&1)
    );
}

#[test]
#[ignore = "expensive boundary patch sizing timing diagnostic"]
fn boundary_patch_sizing_stage_timings_are_observable() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(1.0);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let sizing = MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [1.0, 0.5, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [0.5, 0.0, 0.5],
                target_size_m: 0.25,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let geometry = cube_geometry();
    let topology = extract_source_topology(&geometry).expect("topology");
    let base_target_size_m = target_size_for_mesh(&topology, &options);
    let effective_target_size_m =
        solid_sizing_target_size(base_target_size_m, &sizing, &options, Some(&topology));
    let mut effective_options = options.clone();
    if effective_target_size_m < base_target_size_m {
        effective_options.target_size = MeshTargetSize::LengthM(effective_target_size_m);
    }
    log_preparation_stage_timings(
        "boundary_patch",
        &geometry,
        &effective_options,
        Some(&sizing),
    );
}

#[test]
fn thin_low_face_topology_uses_bounded_surface_options() {
    let geometry = thin_box_geometry();
    let topology = extract_source_topology(&geometry).expect("topology");

    assert!(thin_low_face_topology(&topology));
    assert_eq!(
        surface_options_for_mesh(&topology).max_curve_segments_per_edge,
        20
    );
}

#[test]
fn non_box_topology_is_not_thin_low_face_topology() {
    let geometry = faceted_cylinder_geometry();
    let topology = extract_source_topology(&geometry).expect("topology");

    assert!(constrained_recovery_topology(&topology));
    assert!(!thin_low_face_topology(&topology));
}

#[test]
fn thin_box_generates_native_structured_box_tet_mesh() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.1_f64.cbrt() / 2.0);

    let preparation =
        prepare_solid_mesh(&thin_box_geometry(), &options).expect("thin mesh prepares");
    let mesh = analysis_mesh_from_preparation(&preparation, &options, None)
        .expect("thin analysis mesh should build");

    assert_eq!(preparation.solver_tet_mesh.elements.len(), 6);
    assert_eq!(preparation.tet_stage.volume_component_count, 1);
    assert_eq!(volume_component_count(&mesh), 1);
}

#[test]
fn solid_boundary_faces_map_element_scalars_for_visualization() {
    let options = VolumeMeshingOptions {
        target_size: MeshTargetSize::LengthM(0.5),
        ..VolumeMeshingOptions::default()
    };
    let mesh = generate_solid_analysis_mesh(&cube_geometry(), &options)
        .expect("solid mesh should generate");
    let element_values = (0..mesh.volume_elements.len())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();

    let mapped = map_volume_scalar_field_to_boundary_faces(&mesh, &element_values)
        .expect("boundary faces should map element scalars");

    assert_eq!(mapped.len(), mesh.boundary_faces.len());
    assert!(mapped.iter().all(|value| value.value.is_finite()));
    assert!(mapped
        .iter()
        .zip(mesh.boundary_faces.iter())
        .all(|(value, face)| value.face_id == face.face_id));
}

#[test]
fn solid_boundary_topology_maps_nodal_vectors_for_visualization() {
    let options = VolumeMeshingOptions {
        target_size: MeshTargetSize::LengthM(0.5),
        ..VolumeMeshingOptions::default()
    };
    let mesh = generate_solid_analysis_mesh(&cube_geometry(), &options)
        .expect("solid mesh should generate");
    let node_values = mesh
        .nodes
        .iter()
        .map(|node| {
            [
                node.coordinates_m[0] * 1.0e-3,
                node.coordinates_m[1] * 2.0e-3,
                node.coordinates_m[2] * 3.0e-3,
            ]
        })
        .collect::<Vec<_>>();

    let boundary_node_values = map_nodal_vector_field_to_boundary_nodes(&mesh, &node_values)
        .expect("boundary nodes should map nodal vectors");
    let boundary_face_values = map_nodal_vector_field_to_boundary_faces(&mesh, &node_values)
        .expect("boundary faces should map nodal vectors");

    let boundary_node_count = mesh
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<BTreeSet<_>>()
        .len();
    assert_eq!(boundary_node_values.len(), boundary_node_count);
    assert_eq!(boundary_face_values.len(), mesh.boundary_faces.len());
    assert!(boundary_node_values
        .iter()
        .flat_map(|value| value.value)
        .all(f64::is_finite));
    assert!(boundary_face_values
        .iter()
        .zip(mesh.boundary_faces.iter())
        .all(|(value, face)| value.face_id == face.face_id));
    assert!(boundary_face_values
        .iter()
        .flat_map(|value| value.value)
        .all(f64::is_finite));
}

fn volume_element_centroid_count_within(
    mesh: &AnalysisMeshArtifact,
    point_m: [f64; 3],
    radius_m: f64,
) -> usize {
    let radius_squared_m = radius_m * radius_m;
    mesh.volume_elements
        .iter()
        .filter(|element| {
            let Some(centroid) = analysis_volume_element_centroid(mesh, element) else {
                return false;
            };
            distance_squared(centroid, point_m) <= radius_squared_m
        })
        .count()
}

fn volume_element_count_with_node_within(
    mesh: &AnalysisMeshArtifact,
    point_m: [f64; 3],
    radius_m: f64,
) -> usize {
    let radius_squared_m = radius_m * radius_m;
    let near_node_ids = mesh
        .nodes
        .iter()
        .filter_map(|node| {
            (distance_squared(node.coordinates_m, point_m) <= radius_squared_m)
                .then_some(node.node_id)
        })
        .collect::<BTreeSet<_>>();
    mesh.volume_elements
        .iter()
        .filter(|element| {
            element
                .node_ids
                .iter()
                .any(|node_id| near_node_ids.contains(node_id))
        })
        .count()
}

fn analysis_volume_element_centroid(
    mesh: &AnalysisMeshArtifact,
    element: &AnalysisVolumeElement,
) -> Option<[f64; 3]> {
    if element.node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0; 3];
    for node_id in &element.node_ids {
        let node = mesh.nodes.iter().find(|node| node.node_id == *node_id)?;
        centroid[0] += node.coordinates_m[0];
        centroid[1] += node.coordinates_m[1];
        centroid[2] += node.coordinates_m[2];
    }
    let scale = 1.0 / element.node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

#[test]
fn solid_sizing_maps_requested_ids_after_skipped_samples() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.75);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let sizing = MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [0.10, 0.10, 0.10],
                target_size_m: f64::NAN,
                reason: Some("invalid".to_string()),
            },
            SizingSample {
                position_m: [0.25, 0.25, 0.25],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let mesh = generate_solid_analysis_mesh_with_sizing(&cube_geometry(), &options, &sizing)
        .expect("solid mesh should generate with a skipped sample");

    assert!(mesh.sizing.applied_samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("structural.load_regions")
            && sample.inserted_breakpoint_count > 0
    }));
    assert!(mesh.sizing.rejected_samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("invalid") && sample.status == "skipped_invalid"
    }));
}

#[test]
fn solid_sizing_reports_unaccepted_requested_samples_as_rejections() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.75);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let sizing = MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [0.25, 0.25, 0.25],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [2.0, 2.0, 2.0],
                target_size_m: 0.25,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let mesh = generate_solid_analysis_mesh_with_sizing(&cube_geometry(), &options, &sizing)
        .expect("solid mesh should generate with a rejected requested sample");
    let evidence =
        crate::build_mesh_evidence_artifact(&mesh, &AnalysisMeshValidationOptions::default());

    assert!(mesh.sizing.applied_samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("structural.load_regions")
            && sample.inserted_breakpoint_count > 0
    }));
    assert!(mesh.sizing.rejected_samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("structural.constraint_regions")
            && sample.status == "rejected_by_recovery"
    }));
    assert_eq!(
        evidence
            .sizing
            .rejected_by_reason
            .get("structural.constraint_regions"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .sizing
            .rejected_by_status
            .get("rejected_by_recovery"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .sizing
            .requested_tet_refinement_rejected_by_reason
            .get("outside_volume"),
        Some(&1)
    );
}

#[test]
fn solid_sizing_bounds_requested_sample_ids_to_seed_budget() {
    let mut samples = Vec::<SizingSample>::new();
    for index in 0..17 {
        samples.push(SizingSample {
            position_m: [
                0.10 + 0.02 * index as f64,
                0.25 + 0.01 * index as f64,
                0.25 + 0.005 * index as f64,
            ],
            target_size_m: 0.25,
            reason: Some(format!("requested.marker.{index}")),
        });
    }
    let sizing = MeshSizingField {
        samples,
        ..MeshSizingField::default()
    };

    let topology = extract_source_topology(&cube_geometry()).expect("cube topology should extract");
    let requested_ids = requested_sizing_sample_ids(&topology, &sizing);

    assert_eq!(requested_ids.len(), 16);
    assert_eq!(requested_ids.get(&0), Some(&0));
    assert_eq!(requested_ids.get(&15), Some(&15));
    assert_eq!(requested_ids.get(&16), None);
}

#[test]
fn solid_sizing_includes_cad_curvature_samples() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.5);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = true;

    let mesh = generate_solid_analysis_mesh(&cube_geometry_with_curvature_evaluator(), &options)
        .expect("solid mesh should generate with CAD curvature sizing");

    assert!(mesh.backend.cad_curvature_query_count > 0);
    assert!(mesh.backend.cad_max_curvature_estimate_1_per_m > 0.0);
    assert!(mesh
        .sizing
        .samples
        .iter()
        .any(|sample| sample.reason.as_deref() == Some("cad.curvature")
            && sample.target_size_m < 0.5));
    let evidence = crate::evidence::build_mesh_evidence_artifact(
        &mesh,
        &AnalysisMeshValidationOptions::default(),
    );
    assert!(evidence.sizing.generated_cad_sample_count > 0);
    assert_eq!(
        evidence.sizing.generated_cad_by_reason.get("cad.curvature"),
        Some(
            &mesh
                .sizing
                .samples
                .iter()
                .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
                .count()
        )
    );
    assert_eq!(
        evidence
            .sizing
            .applied_by_reason
            .get("cad.curvature")
            .copied()
            .unwrap_or_default(),
        mesh.sizing
            .applied_samples
            .iter()
            .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
            .count()
    );
    let curvature_inserted = evidence
        .sizing
        .inserted_breakpoint_by_reason
        .get("cad.curvature")
        .copied()
        .unwrap_or_default();
    let curvature_uninserted = evidence
        .sizing
        .uninserted_sample_by_reason
        .get("cad.curvature")
        .copied()
        .unwrap_or_default();
    assert_eq!(
        curvature_inserted + curvature_uninserted,
        mesh.sizing
            .applied_samples
            .iter()
            .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
            .count()
    );
    assert_eq!(
        mesh.backend.tet_rejected_requested_refinement_point_count,
        0
    );
}

#[test]
fn solid_preparation_consumes_live_cad_provider_for_surface_and_sizing() {
    #[derive(Debug)]
    struct LiveProvider;

    impl CadFaceEvaluatorProvider for LiveProvider {
        fn evaluate_face(
            &self,
            request: &crate::cad::eval::CadFaceEvaluationRequest<'_>,
        ) -> Vec<CadFaceEvaluationSample> {
            assert_eq!(request.imported_face_id, Some(1));
            assert_eq!(request.evaluator_id, Some("cad_face_1"));
            assert!(request.supports_projection);
            assert!(request.supports_derivatives);
            assert!(request.supports_curvature);
            let points = if request.source_face_id == 2 {
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
            } else {
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
            };
            points
                .into_iter()
                .enumerate()
                .map(|(index, point)| CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: point,
                    uv: Some([point[0], point[1]]),
                    projected_point_m: Some(point),
                    unit_normal: Some(match index {
                        0 => [0.0, 0.0, 1.0],
                        1 => [0.0, 0.8, 0.6],
                        _ => [0.8, 0.0, 0.6],
                    }),
                    projection_error_m: Some(0.0),
                })
                .collect()
        }
    }

    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.5);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = true;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let mut geometry = cube_geometry_with_curvature_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0]
        .evaluation_samples
        .clear();

    let preparation = prepare_solid_mesh_with_cad_evaluator(&geometry, &options, &LiveProvider)
        .expect("solid preparation should consume live CAD provider");
    let mesh = analysis_mesh_from_preparation(&preparation, &options, None)
        .expect("solid mesh should preserve live CAD evidence");
    let evidence = crate::build_mesh_evidence_artifact(
        &mesh,
        &solid_validation_options(&preparation, &options),
    );
    let curvature_samples = preparation
        .effective_sizing
        .as_ref()
        .expect("live curvature should create sizing")
        .samples
        .iter()
        .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
        .collect::<Vec<_>>();

    assert_eq!(
        preparation.cad_evaluation.source,
        CadEvaluationSource::ParametricCad
    );
    assert!(preparation.cad_evaluation_report.live_query_face_count > 0);
    assert_eq!(
        preparation.cad_evaluation_report.live_query_face_count,
        preparation.cad_evaluation_report.exact_query_face_count
    );
    assert_eq!(
        preparation
            .cad_evaluation_report
            .projection_supported_face_count,
        preparation.cad_evaluation_report.live_query_face_count
    );
    assert_eq!(
        preparation
            .cad_evaluation_report
            .normal_supported_face_count,
        preparation.cad_evaluation_report.live_query_face_count
    );
    assert_eq!(
        preparation
            .cad_evaluation_report
            .derivative_supported_face_count,
        preparation.cad_evaluation_report.live_query_face_count
    );
    assert_eq!(
        preparation
            .cad_evaluation_report
            .curvature_supported_face_count,
        preparation.cad_evaluation_report.live_query_face_count
    );
    assert_eq!(
        preparation
            .cad_evaluation_report
            .missing_exact_query_face_count,
        0
    );
    assert_eq!(
        preparation
            .cad_evaluation_report
            .missing_derivative_query_face_count,
        0
    );
    assert_eq!(
        preparation
            .cad_evaluation_report
            .missing_curvature_query_face_count,
        0
    );
    assert!(preparation.cad_evaluation_report.derivative_query_count > 0);
    assert!(preparation.cad_evaluation_report.curvature_query_count > 0);
    assert!(preparation.cad_evaluation_report.uv_domain_face_count > 0);
    assert!(preparation
        .surface
        .elements
        .iter()
        .any(|element| element.cad_face_id.is_some()));
    assert!(!curvature_samples.is_empty());
    assert!(curvature_samples
        .iter()
        .all(|sample| sample.target_size_m < 0.5));
    assert_eq!(mesh.backend.cad_evaluation_source, "parametric_cad");
    assert_eq!(
        mesh.backend.cad_evaluation_live_query_face_count,
        preparation.cad_evaluation_report.live_query_face_count
    );
    assert_eq!(
        mesh.backend.cad_evaluation_projection_supported_face_count,
        preparation
            .cad_evaluation_report
            .projection_supported_face_count
    );
    assert_eq!(
        mesh.backend.cad_evaluation_curvature_supported_face_count,
        preparation
            .cad_evaluation_report
            .curvature_supported_face_count
    );
    assert_eq!(
        mesh.backend.cad_evaluation_missing_exact_query_face_count,
        0
    );
    assert_eq!(
        mesh.backend
            .cad_evaluation_missing_derivative_query_face_count,
        0
    );
    assert_eq!(
        mesh.backend
            .cad_evaluation_missing_curvature_query_face_count,
        0
    );
    assert_eq!(
        mesh.backend.cad_uv_domain_face_count,
        preparation.cad_evaluation_report.uv_domain_face_count
    );
    assert_eq!(
        mesh.backend.cad_uv_projection_out_of_bounds_count,
        preparation
            .cad_evaluation_report
            .uv_projection_out_of_bounds_count
    );
    assert_eq!(
        mesh.backend.surface_cad_face_count,
        surface_cad_face_count(&preparation.surface)
    );
    assert_eq!(
        mesh.backend.surface_exact_cad_sample_node_count,
        preparation.surface.exact_cad_sample_node_count
    );
    assert_eq!(
        mesh.backend.surface_rejected_exact_cad_sample_count,
        preparation.surface.rejected_exact_cad_sample_count
    );
    assert_eq!(
        mesh.backend.surface_max_cad_projection_error_m,
        surface_max_cad_projection_error_m(&preparation.surface)
    );
    assert_eq!(
        evidence.cad.evaluation_source,
        mesh.backend.cad_evaluation_source
    );
    assert_eq!(
        evidence.cad.live_query_face_count,
        preparation.cad_evaluation_report.live_query_face_count
    );
    assert_eq!(
        evidence.cad.exact_query_face_count,
        preparation.cad_evaluation_report.exact_query_face_count
    );
    assert_eq!(
        evidence.cad.missing_derivative_query_face_count,
        preparation
            .cad_evaluation_report
            .missing_derivative_query_face_count
    );
    assert_eq!(
        evidence.cad.missing_curvature_query_face_count,
        preparation
            .cad_evaluation_report
            .missing_curvature_query_face_count
    );
    assert_eq!(
        evidence.cad.projection_supported_face_count,
        preparation
            .cad_evaluation_report
            .projection_supported_face_count
    );
    assert_eq!(
        evidence.cad.curvature_supported_face_count,
        preparation
            .cad_evaluation_report
            .curvature_supported_face_count
    );
    assert_eq!(
        evidence.cad.surface_exact_cad_sample_node_count,
        preparation.surface.exact_cad_sample_node_count
    );
    assert_eq!(
        evidence.cad.surface_rejected_exact_cad_sample_count,
        preparation.surface.rejected_exact_cad_sample_count
    );
    assert_eq!(
        evidence.cad.surface_max_projection_error_m,
        surface_max_cad_projection_error_m(&preparation.surface)
    );
    assert!(mesh
        .sizing
        .samples
        .iter()
        .any(|sample| sample.reason.as_deref() == Some("cad.curvature")));
}

#[test]
fn solid_sizing_includes_small_feature_edge_and_proximity_samples() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.5);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = true;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;

    let geometry = thin_box_geometry();
    let topology = extract_source_topology(&geometry).expect("topology");
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
    let cad_evaluation = crate::cad::eval::build_cad_evaluation_model(&cad_topology, &topology)
        .expect("cad evaluation");
    let sizing = solid_effective_sizing(&topology, &cad_evaluation, &options, None)
        .expect("feature-edge sizing");

    assert!(sizing.samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("cad.feature_edge") && sample.target_size_m < 0.1
    }));
    assert!(sizing.samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("cad.proximity") && sample.target_size_m < 0.1
    }));
    assert_eq!(
        requested_refinement_selection(&topology, Some(&sizing)).count,
        0
    );
}

#[test]
#[ignore = "expensive thin-feature stage timing diagnostic"]
fn thin_box_preparation_stage_timings_are_observable() {
    let geometry = thin_box_geometry();
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.1_f64.cbrt() / 2.0);
    options.max_elements = 50_000;
    log_preparation_stage_timings("thin_box", &geometry, &options, None);
}

#[test]
#[ignore = "expensive faceted fixture stage timing diagnostic"]
fn faceted_cylinder_preparation_stage_timings_are_observable() {
    let geometry = faceted_cylinder_geometry();
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.459_259_458_684_314_6);
    options.max_elements = 50_000;
    log_preparation_stage_timings("faceted_cylinder", &geometry, &options, None);
}

#[test]
#[ignore = "expensive faceted fixture generation timing diagnostic"]
fn faceted_cylinder_generation_timing_is_observable() {
    let geometry = faceted_cylinder_geometry();
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.459_259_458_684_314_6);
    options.max_elements = 50_000;
    let started = std::time::Instant::now();
    match generate_solid_analysis_mesh(&geometry, &options) {
        Ok(mesh) => eprintln!(
            "faceted_cylinder generation elapsed_ms={:.1} nodes={} elements={}",
            started.elapsed().as_secs_f64() * 1000.0,
            mesh.nodes.len(),
            mesh.volume_elements.len()
        ),
        Err(err) => eprintln!(
            "faceted_cylinder generation_failed elapsed_ms={:.1}: {err}",
            started.elapsed().as_secs_f64() * 1000.0
        ),
    }
}

#[test]
fn faceted_cylinder_fails_closed_until_general_plc_tet_generation_exists() {
    let geometry = faceted_cylinder_geometry();
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.459_259_458_684_314_6);
    options.max_elements = 50_000;

    assert_eq!(
        prepare_solid_mesh(&geometry, &options),
        Err(SolidMeshError::TetGeneration(
            TetGenerationError::UnsupportedStructuredBoxPlc
        ))
    );
}

fn log_preparation_stage_timings(
    label: &str,
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
) {
    let started = std::time::Instant::now();
    let topology = extract_source_topology(geometry).expect("topology");
    eprintln!(
        "{label} topology elapsed_ms={:.1} vertices={} edges={} faces={} constrained={}",
        started.elapsed().as_secs_f64() * 1000.0,
        topology.vertices.len(),
        topology.edges.len(),
        topology.faces.len(),
        constrained_recovery_topology(&topology)
    );

    let stage = std::time::Instant::now();
    let cad_topology = build_cad_topology(geometry, &topology).expect("cad topology");
    eprintln!(
        "{label} cad_topology elapsed_ms={:.1}",
        stage.elapsed().as_secs_f64() * 1000.0
    );

    let stage = std::time::Instant::now();
    let cad_evaluation = crate::cad::eval::build_cad_evaluation_model(&cad_topology, &topology)
        .expect("cad evaluation");
    eprintln!(
        "{label} cad_evaluation elapsed_ms={:.1}",
        stage.elapsed().as_secs_f64() * 1000.0
    );

    let stage = std::time::Instant::now();
    let _effective_sizing = solid_effective_sizing(&topology, &cad_evaluation, options, sizing);
    eprintln!(
        "{label} sizing elapsed_ms={:.1}",
        stage.elapsed().as_secs_f64() * 1000.0
    );

    let stage = std::time::Instant::now();
    let curves = discretize_topology_curves(&topology, curve_options_for_mesh(&topology, options))
        .expect("curves");
    eprintln!(
        "{label} curves elapsed_ms={:.1} nodes={} elements={}",
        stage.elapsed().as_secs_f64() * 1000.0,
        curves.nodes.len(),
        curves.elements.len()
    );

    let stage = std::time::Instant::now();
    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        surface_options_for_mesh(&topology),
    )
    .expect("surface");
    eprintln!(
        "{label} surface elapsed_ms={:.1} nodes={} elements={}",
        stage.elapsed().as_secs_f64() * 1000.0,
        surface.nodes.len(),
        surface.elements.len()
    );

    let stage = std::time::Instant::now();
    let protected_boundary_complex =
        build_protected_boundary_complex(&surface).expect("protected boundary complex");
    eprintln!(
        "{label} plc elapsed_ms={:.1} nodes={} facets={} protected_edges={}",
        stage.elapsed().as_secs_f64() * 1000.0,
        protected_boundary_complex.nodes.len(),
        protected_boundary_complex.facets.len(),
        protected_boundary_complex.protected_edges.len()
    );

    let stage = std::time::Instant::now();
    let solver_tet_mesh = generate_structured_box_tet_mesh_from_plc(&protected_boundary_complex)
        .expect("structured box Tet mesh");
    eprintln!(
        "{label} tet elapsed_ms={:.1} nodes={} elements={} boundary_faces={}",
        stage.elapsed().as_secs_f64() * 1000.0,
        solver_tet_mesh.nodes.len(),
        solver_tet_mesh.elements.len(),
        solver_tet_mesh.boundary_faces.len()
    );
}

fn thin_axis_cap_element_count(surface: &SurfaceDiscretization) -> usize {
    if surface.nodes.is_empty() {
        return 0;
    }
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for node in &surface.nodes {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(node.coordinates_m[axis]);
            bounds_max[axis] = bounds_max[axis].max(node.coordinates_m[axis]);
        }
    }
    let Some(thin_axis) = (0..3)
        .filter(|axis| (bounds_max[*axis] - bounds_min[*axis]).is_finite())
        .min_by(|left, right| {
            (bounds_max[*left] - bounds_min[*left])
                .total_cmp(&(bounds_max[*right] - bounds_min[*right]))
        })
    else {
        return 0;
    };
    surface
        .elements
        .iter()
        .filter(|element| {
            element.node_ids.iter().all(|node_id| {
                let coordinate = surface.nodes[*node_id as usize].coordinates_m[thin_axis];
                (coordinate - bounds_min[thin_axis]).abs() <= 1.0e-9
                    || (coordinate - bounds_max[thin_axis]).abs() <= 1.0e-9
            })
        })
        .count()
}

#[test]
fn solid_sizing_consumes_valid_anisotropic_samples_conservatively() {
    let mut options = VolumeMeshingOptions::default();
    options.target_size = MeshTargetSize::LengthM(0.5);
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    let sizing = MeshSizingField {
        anisotropic_samples: vec![
            AnisotropicSizingSample {
                position_m: [0.25, 0.25, 0.25],
                target_sizes_m: [0.05, 0.2, 0.4],
                directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                reason: Some("boundary_layer".to_string()),
            },
            AnisotropicSizingSample {
                position_m: [0.75, 0.75, 0.75],
                target_sizes_m: [0.05, -0.2, 0.4],
                directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                reason: Some("cad.proximity".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };
    let geometry = cube_geometry();
    let topology = extract_source_topology(&geometry).expect("topology");
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
    let cad_evaluation = crate::cad::eval::build_cad_evaluation_model(&cad_topology, &topology)
        .expect("cad evaluation");

    let effective = solid_effective_sizing(&topology, &cad_evaluation, &options, Some(&sizing))
        .expect("anisotropic sizing should produce an effective sizing field");

    assert_eq!(effective.anisotropic_samples.len(), 2);
    assert!(effective.samples.iter().any(|sample| {
        sample.reason.as_deref() == Some("boundary_layer")
            && sample.position_m == [0.25, 0.25, 0.25]
            && (sample.target_size_m - 0.05).abs() <= 1.0e-12
    }));
    assert!(!effective
        .samples
        .iter()
        .any(|sample| sample.position_m == [0.75, 0.75, 0.75]));
    assert_eq!(
        requested_refinement_selection(&topology, Some(&effective)).count,
        1
    );
    assert_eq!(solid_sizing_target_size(0.5, &sizing, &options, None), 0.05);
}

#[test]
fn solid_point_sizing_samples_do_not_lower_global_target_size() {
    let options = VolumeMeshingOptions {
        target_size: MeshTargetSize::LengthM(1.0),
        ..VolumeMeshingOptions::default()
    };
    let sizing = MeshSizingField {
        samples: vec![SizingSample {
            position_m: [1.0, 0.5, 0.5],
            target_size_m: 0.1,
            reason: Some("structural.load_regions".to_string()),
        }],
        ..MeshSizingField::default()
    };

    let geometry = cube_geometry();
    let topology = extract_source_topology(&geometry).expect("topology");

    assert_eq!(
        solid_sizing_target_size(1.0, &sizing, &options, Some(&topology)),
        1.0
    );
}

#[test]
fn solid_physics_sizing_samples_can_lower_global_target_size() {
    let options = VolumeMeshingOptions {
        target_size: MeshTargetSize::LengthM(1.0),
        ..VolumeMeshingOptions::default()
    };
    let sizing = MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.25, 0.25, 0.25],
            target_size_m: 0.25,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    };

    assert_eq!(solid_sizing_target_size(1.0, &sizing, &options, None), 0.25);
}

#[test]
fn solid_sizing_includes_interface_samples_by_focus_level() {
    let mut options = VolumeMeshingOptions::default();
    options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
    options.refinement.focus.curvature = false;
    options.refinement.focus.small_features = false;
    options.refinement.focus.interfaces = RefinementFocusLevel::Normal;
    let geometry = cube_geometry();
    let topology = extract_source_topology(&geometry).expect("topology");
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
    let cad_evaluation = crate::cad::eval::build_cad_evaluation_model(&cad_topology, &topology)
        .expect("cad evaluation");

    let normal = solid_effective_sizing(&topology, &cad_evaluation, &options, None)
        .expect("normal interface sizing");
    let normal_targets = normal
        .samples
        .iter()
        .filter(|sample| sample.reason.as_deref() == Some("cad.interface"))
        .map(|sample| sample.target_size_m)
        .collect::<Vec<_>>();
    assert!(!normal_targets.is_empty());
    assert!(normal_targets
        .iter()
        .all(|target| (*target - 0.5).abs() < 1.0e-12));

    options.refinement.focus.interfaces = RefinementFocusLevel::Fine;
    let fine = solid_effective_sizing(&topology, &cad_evaluation, &options, None)
        .expect("fine interface sizing");
    let fine_targets = fine
        .samples
        .iter()
        .filter(|sample| sample.reason.as_deref() == Some("cad.interface"))
        .map(|sample| sample.target_size_m)
        .collect::<Vec<_>>();
    assert_eq!(fine_targets.len(), normal_targets.len());
    assert!(fine_targets
        .iter()
        .all(|target| (*target - 0.25).abs() < 1.0e-12));

    options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    assert!(solid_effective_sizing(&topology, &cad_evaluation, &options, None).is_none());
}

#[test]
fn quality_report_summarizes_boundary_projection_error() {
    let quality = quality_report(
        vec![ElementQuality {
            element_id: "tet_1".to_string(),
            scaled_jacobian: 0.5,
            exact_scaled_jacobian: 0.45,
            aspect_ratio: 2.0,
            volume_m3: 1.0,
        }],
        [2.0e-6, f64::NAN, 4.0e-6],
    );

    assert_eq!(quality.mean_boundary_projection_error_m, 3.0e-6);
    assert_eq!(quality.max_boundary_projection_error_m, 4.0e-6);
}

#[test]
fn solid_validation_rejects_surface_projection_error() {
    let mut preparation = prepare_solid_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
        .expect("solid preparation should run");
    preparation.surface.elements[0].max_projection_error_m = 2.0e-6;

    let mut options = VolumeMeshingOptions::default();
    options.validation.quality.max_boundary_projection_error_m = 1.0e-6;
    let err = analysis_mesh_from_preparation(&preparation, &options, None)
        .expect_err("projection error should fail solid validation");

    assert!(matches!(
        err,
        SolidMeshError::Validation(
            crate::AnalysisMeshValidationError::QualityThresholdFailed { reason }
        ) if reason == "max_boundary_projection_error_m"
    ));
}

fn cube_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_solid_cube".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_cube.step".to_string(),
            sha256: "generic-cube".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "cube_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 8,
            element_count: 12,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "cube_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
        )],
        regions: vec![
            Region {
                region_id: "root".to_string(),
                name: "root".to_string(),
                tag: None,
                cad_ownership: None,
            },
            Region {
                region_id: "tip".to_string(),
                name: "tip".to_string(),
                tag: None,
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "root",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(0, 6)],
            ),
            RegionEntityMapping::new(
                "tip",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(6, 6)],
            ),
        ],
        diagnostics: Vec::new(),
    }
}

fn cube_geometry_with_curvature_evaluator() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.regions.push(Region {
        region_id: "curved_face".to_string(),
        name: "curved_face".to_string(),
        tag: Some("cad_face".to_string()),
        cad_ownership: Some(CadRegionOwnership {
            face_id: Some(1),
            label: Some(CadLabelRef {
                label_entry: "0:1:1".to_string(),
                name: "curved_face".to_string(),
                kind: CadSemanticKind::Face,
            }),
            owner_path: Vec::new(),
            layers: Vec::new(),
            color: None,
            material: None,
        }),
    });
    geometry
        .region_entity_mappings
        .push(RegionEntityMapping::new(
            "curved_face",
            "cube_surface",
            EntityKind::Face,
            vec![EntityIdRange::new(2, 2)],
        ));
    geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
        evaluator_id: "cad_evaluator_test".to_string(),
        backend: "test".to_string(),
        format_name: "step".to_string(),
        requires_source_geometry: true,
        faces: vec![CadFaceEvaluator {
            evaluator_id: "cad_face_1".to_string(),
            imported_face_id: 1,
            name: "curved_face".to_string(),
            supports_point_evaluation: true,
            supports_projection: true,
            supports_normal: true,
            supports_derivatives: true,
            supports_curvature: true,
            reference_point_m: Some([0.5, 0.5, 1.0]),
            reference_unit_normal: Some([0.0, 0.0, 1.0]),
            evaluation_samples: vec![
                CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [0.0, 0.0, 1.0],
                    uv: Some([0.0, 0.0]),
                    projected_point_m: Some([0.0, 0.0, 1.0]),
                    unit_normal: Some([0.0, 0.0, 1.0]),
                    projection_error_m: Some(0.0),
                },
                CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [1.0, 0.0, 1.0],
                    uv: Some([1.0, 0.0]),
                    projected_point_m: Some([1.0, 0.0, 1.0]),
                    unit_normal: Some([0.0, 0.9, 0.4358898943540673]),
                    projection_error_m: Some(0.0),
                },
                CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [0.0, 1.0, 1.0],
                    uv: Some([0.0, 1.0]),
                    projected_point_m: Some([0.0, 1.0, 1.0]),
                    unit_normal: Some([0.9, 0.0, 0.4358898943540673]),
                    projection_error_m: Some(0.0),
                },
            ],
        }],
        curves: Vec::new(),
    }];
    geometry
}

fn thin_box_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_solid_thin_box".to_string();
    geometry.source.sha256 = "generic-thin-box".to_string();
    if let Some(surface) = geometry.surface_meshes.first_mut() {
        for vertex in &mut surface.vertices {
            if vertex[2] > 0.0 {
                vertex[2] = 0.1;
            }
        }
    }
    geometry
}

fn faceted_cylinder_geometry() -> GeometryAsset {
    let segment_count = 16_usize;
    let radius_m = 0.5_f64;
    let height_m = 1.0_f64;
    let mut vertices = Vec::<[f64; 3]>::with_capacity(segment_count * 2 + 2);
    for z in [0.0, height_m] {
        for index in 0..segment_count {
            let theta = std::f64::consts::TAU * index as f64 / segment_count as f64;
            vertices.push([radius_m * theta.cos(), radius_m * theta.sin(), z]);
        }
    }
    let bottom_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, 0.0]);
    let top_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, height_m]);

    let mut triangles = Vec::<[u32; 3]>::with_capacity(segment_count * 4);
    let top_offset = segment_count as u32;
    for index in 0..segment_count as u32 {
        let next = (index + 1) % segment_count as u32;
        triangles.push([index, next, top_offset + next]);
        triangles.push([index, top_offset + next, top_offset + index]);
        triangles.push([bottom_center, next, index]);
        triangles.push([top_center, top_offset + index, top_offset + next]);
    }
    let face_count = triangles.len() as u64;
    GeometryAsset {
        geometry_id: "geo_solid_faceted_cylinder".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_faceted_cylinder.step".to_string(),
            sha256: "generic-faceted-cylinder".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "faceted_cylinder_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: vertices.len() as u64,
            element_count: face_count,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "faceted_cylinder_surface",
            vertices,
            triangles,
        )],
        regions: vec![
            Region {
                region_id: "root".to_string(),
                name: "root".to_string(),
                tag: None,
                cad_ownership: None,
            },
            Region {
                region_id: "tip".to_string(),
                name: "tip".to_string(),
                tag: None,
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "root",
                "faceted_cylinder_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(0, face_count / 2)],
            ),
            RegionEntityMapping::new(
                "tip",
                "faceted_cylinder_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(
                    face_count / 2,
                    face_count - face_count / 2,
                )],
            ),
        ],
        diagnostics: Vec::new(),
    }
}
