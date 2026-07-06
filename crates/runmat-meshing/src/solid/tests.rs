use super::*;
use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    CadCurveEvaluationSample, CadCurveEvaluationSampleSource, CadCurveEvaluator, CadEvaluatorSet,
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};
use runmat_meshing_cad::SourceTopologyError;
use runmat_meshing_core::contracts::{
    AnalysisBoundaryFace, AnalysisMeshNode, AnalysisVolumeElement,
};
use runmat_meshing_core::fixtures::{
    split_material_through_hole_plate_geometry, through_hole_plate_geometry,
};
use runmat_meshing_core::{
    validate_analysis_mesh, validate_analysis_mesh_with_options, AnalysisFieldTopologyLocation,
    AnalysisMeshValidationOptions, MeshBackendKind, MeshSizingField, MeshTargetSize,
    QualityThresholds, SizingSample, SourceEntityKind, VolumeMeshingOptions,
    ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID, ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
    TETRAHEDRON4_FIELD_ELEMENT_KIND, TRI3_FIELD_ELEMENT_KIND,
};

use crate::visualization::{
    map_nodal_vector_field_to_boundary_faces, map_nodal_vector_field_to_boundary_nodes,
    map_volume_scalar_field_to_boundary_faces,
};

#[test]
fn auto_backend_recovers_plc_constraints_for_cube() {
    let mesh = generate_analysis_mesh(&cube_geometry(), VolumeMeshingOptions::default())
        .expect("auto backend should run the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.backend.algorithm, "plc_tetrahedron/v1");
    assert_eq!(mesh.backend.tetrahedron_generation_family, "structured_box");
    assert!(!mesh.volume_elements.is_empty());
    assert!(!mesh.boundary_faces.is_empty());
    let source_face_count = mesh
        .boundary_faces
        .iter()
        .flat_map(|face| face.provenance.iter())
        .filter(|provenance| provenance.source_entity_kind == SourceEntityKind::Face)
        .map(|provenance| provenance.source_entity_id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let source_edge_count = mesh
        .boundary_edges
        .iter()
        .flat_map(|edge| edge.provenance.iter())
        .filter(|provenance| provenance.source_entity_kind == SourceEntityKind::Edge)
        .map(|provenance| provenance.source_entity_id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    assert_eq!(source_face_count, 6);
    assert_eq!(source_edge_count, 12);
    assert_eq!(mesh.backend.plc_input_protected_edge_count, 12);
    assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
    assert_eq!(
        mesh.backend
            .tetrahedron_missing_source_edge_recovery_item_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_recovered_source_edge_recovery_item_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_missing_source_face_recovery_item_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_recovered_source_face_recovery_item_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_recovered_material_interface_recovery_item_count,
        0
    );
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!(mesh.backend.tetrahedron_min_exact_scaled_jacobian >= 0.15);
    assert_eq!(
        mesh.backend
            .tetrahedron_exact_scaled_jacobian_below_threshold_count,
        0
    );
    assert_eq!(mesh.backend.tetrahedron_optimization_target_seed_count, 0);
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_skipped_target_seed_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_local_reconnection_attempt_count,
        0
    );
    assert_eq!(
        mesh.backend.tetrahedron_optimization_budget_limited_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_local_reconnection_budget_limited_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_interior_smoothing_attempt_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_interior_smoothing_accepted_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_interior_smoothing_rejected_count,
        0
    );
    assert!(mesh
        .backend
        .tetrahedron_optimization_interior_smoothing_rejected_by_reason
        .is_empty());
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_boundary_smoothing_attempt_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_boundary_smoothing_accepted_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_boundary_smoothing_rejected_count,
        0
    );
    assert!(mesh
        .backend
        .tetrahedron_optimization_boundary_smoothing_rejected_by_reason
        .is_empty());
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_local_reconnection_accepted_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_local_reconnection_rejected_count,
        0
    );
    assert!(mesh
        .backend
        .tetrahedron_optimization_local_reconnection_rejected_by_reason
        .is_empty());
    assert!(mesh
        .backend
        .tetrahedron_missing_source_face_recovery_ids
        .is_empty());
    assert!(mesh
        .backend
        .tetrahedron_missing_source_edge_recovery_ids
        .is_empty());
    assert!(mesh
        .backend
        .tetrahedron_missing_material_interface_recovery_ids
        .is_empty());
    assert_eq!(mesh.backend.tetrahedron_sliver_removed_count, 0);
    assert_eq!(
        field_topology_count(
            &mesh,
            ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
            AnalysisFieldTopologyLocation::Node,
            None,
        ),
        Some(mesh.nodes.len())
    );
    assert_eq!(
        field_topology_count(
            &mesh,
            ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
            AnalysisFieldTopologyLocation::VolumeElement,
            Some(TETRAHEDRON4_FIELD_ELEMENT_KIND),
        ),
        Some(mesh.volume_elements.len())
    );
    assert_eq!(
        field_topology_count(
            &mesh,
            ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID,
            AnalysisFieldTopologyLocation::BoundaryFace,
            Some(TRI3_FIELD_ELEMENT_KIND),
        ),
        Some(mesh.boundary_faces.len())
    );
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("default auto cube should be solve-ready after healed CAD face loops");
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            min_boundary_face_recovery_ratio: 1.0,
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("root solid pipeline should recover PLC constraints before quality optimization");
}

fn field_topology_count(
    mesh: &runmat_meshing_core::AnalysisMeshArtifact,
    topology_id: &str,
    location: AnalysisFieldTopologyLocation,
    element_kind: Option<&str>,
) -> Option<usize> {
    mesh.field_topology
        .iter()
        .find(|descriptor| {
            descriptor.topology_id == topology_id
                && descriptor.location == location
                && descriptor.element_kind.as_deref() == element_kind
        })
        .map(|descriptor| descriptor.entity_count)
}

#[test]
fn solid_curve_evaluator_provider_filters_geometry_curve_samples() {
    let mut geometry = cube_geometry();
    geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
        evaluator_id: "cad_evaluator_test".to_string(),
        backend: "test".to_string(),
        format_name: "step".to_string(),
        requires_source_geometry: true,
        faces: Vec::new(),
        curves: vec![CadCurveEvaluator {
            evaluator_id: "cad_curve_12".to_string(),
            imported_curve_id: 12,
            name: "edge".to_string(),
            supports_point_evaluation: true,
            supports_projection: true,
            supports_tangent: true,
            supports_curvature: true,
            evaluation_samples: vec![
                CadCurveEvaluationSample {
                    source: CadCurveEvaluationSampleSource::BackendQuery,
                    parameter: 0.5,
                    point_m: [0.5, 0.1, 0.0],
                    projected_point_m: Some([0.5, 0.2, 0.0]),
                    tangent_m: Some([1.0, 0.0, 0.0]),
                    curvature_1_per_m: Some(0.5),
                    projection_error_m: Some(0.1),
                },
                CadCurveEvaluationSample {
                    source: CadCurveEvaluationSampleSource::BackendQuery,
                    parameter: 0.25,
                    point_m: [0.25, 0.0, 0.0],
                    projected_point_m: None,
                    tangent_m: None,
                    curvature_1_per_m: None,
                    projection_error_m: None,
                },
            ],
        }],
    }];
    let provider = GeometryCadCurveEvaluatorProvider {
        geometry: &geometry,
    };

    let samples = provider.evaluate_curve(&CadCurveEvaluationRequest {
        cad_edge_id: "cad_edge_0",
        source_edge_id: 0,
        imported_curve_id: Some(12),
        evaluator_id: Some("cad_curve_12"),
        supports_point_evaluation: true,
        supports_projection: true,
        supports_tangent: true,
        supports_curvature: true,
        parameters: &[0.5],
    });

    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].parameter, 0.5);
    assert_eq!(samples[0].projected_point_m, Some([0.5, 0.2, 0.0]));
}

#[test]
fn auto_backend_preserves_recovered_material_regions_for_split_cube() {
    let mesh = generate_analysis_mesh(
        &split_material_cube_geometry(),
        VolumeMeshingOptions::default(),
    )
    .expect("split-region cube should recover material ownership into final artifact");

    let material_region_ids = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        material_region_ids,
        BTreeSet::from(["region_base", "region_cap"])
    );
    assert!(mesh
        .volume_elements
        .iter()
        .all(|element| element.material_region_id != "unclassified"));
    assert_eq!(
        mesh.backend
            .tetrahedron_recovered_material_interface_recovery_item_count,
        2
    );
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["region_base".to_string(), "region_cap".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("final artifact should expose both recovered material regions");
}

#[test]
fn auto_backend_generates_thin_arm_solid() {
    let mesh = generate_analysis_mesh(&thin_arm_geometry(), VolumeMeshingOptions::default())
        .expect("thin arm should run through the root solid pipeline");

    let bounds = mesh.nodes.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(mut min, mut max), node| {
            for axis in 0..3 {
                min[axis] = min[axis].min(node.coordinates_m[axis]);
                max[axis] = max[axis].max(node.coordinates_m[axis]);
            }
            (min, max)
        },
    );
    let material_region_ids = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.backend.tetrahedron_generation_family, "structured_box");
    assert_eq!(material_region_ids, BTreeSet::from(["body"]));
    assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!((bounds.0[0] - 0.0).abs() < 1.0e-9);
    assert!((bounds.1[0] - 3.0).abs() < 1.0e-9);
    assert!((bounds.1[1] - 0.75).abs() < 1.0e-9);
    assert!((bounds.1[2] - 0.75).abs() < 1.0e-9);
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["body".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("thin arm solid mesh should be solve-ready");
}

#[test]
fn explicit_sizing_refines_recovered_cube_source_edges() {
    let sizing = MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.5, 0.0, 0.0],
            target_size_m: 0.2,
            reason: Some("feature_edge".to_string()),
        }],
        ..MeshSizingField::default()
    };

    let mesh = generate_analysis_mesh_with_sizing(
        &cube_geometry(),
        VolumeMeshingOptions::default(),
        &sizing,
    )
    .expect("sizing-aware solid pipeline should refine source-edge curves");

    assert_eq!(mesh.backend.backend, "solid");
    assert!(mesh.backend.plc_input_protected_edge_count > 12);
    assert!(mesh.backend.tetrahedron_source_edge_recovery_item_count > 12);
    assert_eq!(
        mesh.backend
            .tetrahedron_missing_source_edge_recovery_item_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_attempted_source_edge_split_refill_item_count,
        0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_applied_source_edge_split_refill_item_count,
        0
    );
    assert_eq!(
        mesh.volume_elements.len(),
        mesh.backend.surface_element_count
    );
    assert!(
        mesh.backend
            .tetrahedron_exact_scaled_jacobian_below_threshold_count
            == 0
    );
    assert_eq!(mesh.backend.tetrahedron_optimization_target_seed_count, 0);
    assert_eq!(
        mesh.backend
            .tetrahedron_optimization_skipped_target_seed_count,
        0
    );
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("refined protected-edge cube should pass the default quality gate");
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("refined protected-edge cube should preserve strict source-edge provenance");
}

#[test]
fn boundary_focus_sizing_records_load_and_constraint_curve_applications() {
    let sizing = MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [0.5, 0.5, 0.0],
                target_size_m: 0.5,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [0.5, 0.5, 1.0],
                target_size_m: 0.5,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    };

    let mesh = generate_analysis_mesh_with_sizing(
        &cube_geometry(),
        VolumeMeshingOptions::default(),
        &sizing,
    )
    .expect("boundary-focus sizing should refine protected solid curves");

    let mut inserted_breakpoints_by_reason = BTreeMap::<&str, usize>::new();
    for application in &mesh.sizing.applied_samples {
        *inserted_breakpoints_by_reason
            .entry(application.reason.as_deref().unwrap_or("unspecified"))
            .or_default() += application.inserted_breakpoint_count;
    }
    let face_domain_application_reasons = mesh
        .sizing
        .applied_samples
        .iter()
        .filter_map(|application| {
            application
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("face-domain sizing samples"))
                .then_some(application.reason.as_deref().unwrap_or("unspecified"))
        })
        .collect::<BTreeSet<_>>();

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.sizing.rejected_samples, Vec::new());
    assert!(
        inserted_breakpoints_by_reason["structural.load_regions"] > 0,
        "load-region sizing should insert protected-edge breakpoints"
    );
    assert!(
        inserted_breakpoints_by_reason["structural.constraint_regions"] > 0,
        "constraint-region sizing should insert protected-edge breakpoints"
    );
    assert!(face_domain_application_reasons.contains("structural.load_regions"));
    assert!(face_domain_application_reasons.contains("structural.constraint_regions"));
    assert!(mesh.backend.plc_input_protected_edge_count > 12);
    assert_eq!(
        mesh.backend
            .tetrahedron_missing_source_edge_recovery_item_count,
        0
    );
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("boundary-focus refined cube should remain solve-ready with source-edge provenance");
}

#[test]
fn explicit_sizing_generates_solve_ready_single_tetrahedron_mesh() {
    let mesh = generate_analysis_mesh(
        &tetrahedron_geometry(),
        VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(10.0),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("Tetrahedron PLC should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "single_tetrahedron"
    );
    assert_eq!(
        mesh.backend.tetrahedron_generation_attempted_family_count,
        3
    );
    assert_eq!(mesh.backend.tetrahedron_generation_rejected_family_count, 2);
    assert_eq!(mesh.backend.tetrahedron_generation_selected_family_index, 3);
    assert_eq!(mesh.volume_elements.len(), 1);
    assert_eq!(mesh.boundary_faces.len(), 4);
    assert_eq!(mesh.backend.plc_input_node_count, 4);
    assert_eq!(mesh.backend.plc_input_facet_count, 4);
    assert_eq!(mesh.backend.plc_input_protected_edge_count, 6);
    assert_eq!(mesh.backend.plc_input_boundary_component_count, 1);
    assert_eq!(mesh.backend.plc_input_boundary_component_node_count, 4);
    assert_eq!(mesh.backend.plc_input_max_boundary_component_node_count, 4);
    assert_eq!(mesh.backend.plc_input_surface_boundary_node_count, 4);
    assert!(mesh.backend.plc_input_shell_nesting_classified);
    assert_eq!(mesh.backend.plc_input_outer_shell_count, 1);
    assert_eq!(mesh.backend.plc_input_nested_shell_count, 0);
    assert_eq!(mesh.backend.plc_input_max_shell_nesting_depth, 0);
    assert!(mesh.boundary_faces.iter().all(|face| face
        .provenance
        .iter()
        .any(|provenance| provenance.source_entity_kind == SourceEntityKind::Face)));
    let source_edge_count = mesh
        .boundary_edges
        .iter()
        .flat_map(|edge| edge.provenance.iter())
        .filter(|provenance| provenance.source_entity_kind == SourceEntityKind::Edge)
        .map(|provenance| provenance.source_entity_id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    assert_eq!(
        source_edge_count,
        mesh.backend.plc_input_protected_edge_count
    );
    assert_eq!(mesh.backend.tetrahedron_source_face_recovery_item_count, 4);
    assert_eq!(mesh.backend.tetrahedron_source_edge_recovery_item_count, 6);
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!(mesh.backend.tetrahedron_min_exact_scaled_jacobian >= 0.15);
    assert_eq!(
        mesh.backend
            .tetrahedron_exact_scaled_jacobian_below_threshold_count,
        0
    );
    assert_eq!(mesh.backend.tetrahedron_sliver_count, 0);
    assert_eq!(
        mesh.backend
            .tetrahedron_missing_source_edge_recovery_item_count,
        0
    );
    assert!(mesh
        .backend
        .tetrahedron_missing_source_face_recovery_ids
        .is_empty());
    assert!(mesh
        .backend
        .tetrahedron_missing_source_edge_recovery_ids
        .is_empty());
    assert!(mesh
        .backend
        .tetrahedron_missing_material_interface_recovery_ids
        .is_empty());
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("single Tetrahedron solid mesh should be solve-ready");
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("single Tetrahedron solid mesh should preserve every protected source edge");
}

#[test]
fn explicit_sizing_generates_solve_ready_convex_octahedron_mesh() {
    let mesh = generate_analysis_mesh(
        &octahedron_geometry(),
        VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(10.0),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("convex octahedron PLC should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "convex_polyhedron"
    );
    assert_eq!(mesh.volume_elements.len(), 8);
    assert_eq!(mesh.boundary_faces.len(), 8);
    assert!(
        mesh.backend
            .tetrahedron_generation_interior_support_candidate_count
            > 1
    );
    assert!(
        mesh.backend
            .tetrahedron_generation_interior_support_accepted_count
            <= 1
    );
    assert_eq!(mesh.backend.tetrahedron_source_face_recovery_item_count, 8);
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("convex octahedron solid mesh should be solve-ready");
}

#[test]
fn auto_backend_generates_faceted_cylinder_solid() {
    let mesh = generate_analysis_mesh(
        &faceted_cylinder_geometry(),
        VolumeMeshingOptions::default(),
    )
    .expect("faceted cylinder should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "convex_polyhedron"
    );
    assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!(mesh.backend.tetrahedron_min_exact_scaled_jacobian >= 0.15);
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["body".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("faceted cylinder solid mesh should be solve-ready");
}

#[test]
fn auto_backend_generates_nested_tetrahedron_shell_solid() {
    let mesh = generate_analysis_mesh(
        &nested_tetrahedron_shell_geometry(),
        VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(10.0),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("nested tetrahedron shell solid should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "nested_tetrahedron_shell"
    );
    assert_eq!(
        mesh.backend.tetrahedron_generation_attempted_family_count,
        1
    );
    assert_eq!(mesh.backend.tetrahedron_generation_rejected_family_count, 0);
    assert_eq!(mesh.backend.tetrahedron_generation_selected_family_index, 1);
    assert_eq!(mesh.backend.plc_input_outer_shell_count, 1);
    assert_eq!(mesh.backend.plc_input_nested_shell_count, 1);
    assert_eq!(mesh.backend.plc_input_max_shell_nesting_depth, 1);
    assert!(
        mesh.backend
            .tetrahedron_generation_nested_shell_outer_node_count
            > 0
    );
    assert!(
        mesh.backend
            .tetrahedron_generation_nested_shell_inner_node_count
            > 0
    );
    assert!(
        mesh.backend
            .tetrahedron_generation_nested_shell_refill_boundary_face_count
            > 0
    );
    assert_eq!(
        mesh.backend
            .tetrahedron_generation_nested_shell_boundary_exact_cover_refill_count
            + mesh
                .backend
                .tetrahedron_generation_nested_shell_boundary_centroid_refinement_refill_count
            + mesh
                .backend
                .tetrahedron_generation_nested_shell_barycentric_partition_refill_count,
        1
    );
    assert_eq!(mesh.boundary_faces.len(), 404);
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!(mesh.backend.tetrahedron_min_exact_scaled_jacobian >= 0.15);
}

#[test]
fn auto_backend_generates_star_shaped_dented_corner_solid() {
    let mesh = generate_analysis_mesh(
        &dented_corner_box_geometry(),
        VolumeMeshingOptions::default(),
    )
    .expect("star-shaped dented-corner solid should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "star_shaped_polyhedron"
    );
    assert_eq!(
        mesh.backend.tetrahedron_generation_attempted_family_count,
        6
    );
    assert_eq!(mesh.backend.tetrahedron_generation_rejected_family_count, 5);
    assert_eq!(mesh.backend.tetrahedron_generation_selected_family_index, 6);
    assert!(
        mesh.backend
            .tetrahedron_generation_interior_support_candidate_count
            > 1
    );
    assert!(
        mesh.backend
            .tetrahedron_generation_interior_support_accepted_count
            <= 1
    );
    assert_eq!(
        mesh.volume_elements.len(),
        mesh.backend.surface_element_count
    );
    assert_eq!(
        mesh.backend.tetrahedron_element_count,
        mesh.volume_elements.len()
    );
    assert_eq!(
        mesh.boundary_faces.len(),
        mesh.backend.surface_element_count
    );
    assert_eq!(
        mesh.backend.tetrahedron_source_face_recovery_item_count,
        mesh.backend.surface_element_count
    );
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!(mesh.backend.tetrahedron_min_exact_scaled_jacobian > 0.0);
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("star-shaped dented-corner solid mesh should be solve-ready");
}

#[test]
fn auto_backend_generates_through_hole_solid() {
    let geometry = through_hole_plate_geometry();
    let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
        .expect("through-hole plate solid should run through the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "holed_polyhedron"
    );
    assert!(!mesh.volume_elements.is_empty());
    assert!(!mesh.boundary_faces.is_empty());
    assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
    assert!(mesh.backend.tetrahedron_min_exact_scaled_jacobian >= 0.15);
    let source_edge_count = mesh
        .boundary_edges
        .iter()
        .flat_map(|edge| edge.provenance.iter())
        .filter(|provenance| provenance.source_entity_kind == SourceEntityKind::Edge)
        .map(|provenance| provenance.source_entity_id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    assert_eq!(
        source_edge_count,
        mesh.backend.plc_input_protected_edge_count
    );
    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("through-hole solid mesh should validate");
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["body".to_string()],
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("through-hole solid mesh should expose body material and source-edge provenance");
}

#[test]
fn auto_backend_generates_close_parallel_wall_slot_solid() {
    let geometry = close_parallel_wall_slot_geometry();
    let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
        .expect("close-wall through-slot solid should run through the root solid pipeline");

    let slot_x_min = geometry.surface_meshes[0].vertices[4..8]
        .iter()
        .map(|vertex| vertex[0])
        .fold(f64::INFINITY, f64::min);
    let slot_x_max = geometry.surface_meshes[0].vertices[4..8]
        .iter()
        .map(|vertex| vertex[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let material_region_ids = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "holed_polyhedron"
    );
    assert!((slot_x_max - slot_x_min - 0.2).abs() < 1.0e-9);
    assert_eq!(material_region_ids, BTreeSet::from(["body"]));
    assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!(
        mesh.backend.tetrahedron_min_exact_scaled_jacobian >= 0.15,
        "min exact scaled Jacobian was {}",
        mesh.backend.tetrahedron_min_exact_scaled_jacobian
    );
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["body".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("close-wall through-slot solid mesh should be solve-ready");
}

#[test]
fn auto_backend_preserves_recovered_material_regions_for_through_hole_solid() {
    let geometry = split_material_through_hole_plate_geometry();
    let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
        .expect("split-region through-hole solid should recover material ownership");

    let material_region_ids = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        material_region_ids,
        BTreeSet::from(["region_base", "region_cap"])
    );
    assert!(mesh
        .volume_elements
        .iter()
        .all(|element| element.material_region_id != "unclassified"));
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "holed_polyhedron"
    );
    assert_eq!(mesh.backend.tetrahedron_material_region_count, 2);
    assert_eq!(
        mesh.backend.tetrahedron_unclassified_material_element_count,
        0
    );
    assert!(
        mesh.backend
            .tetrahedron_recovered_material_interface_recovery_item_count
            > 0
    );
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["region_base".to_string(), "region_cap".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("through-hole solid artifact should expose both recovered material regions");
}

#[test]
fn auto_backend_generates_split_material_close_parallel_wall_slot_solid() {
    let geometry = split_material_close_parallel_wall_slot_geometry();
    let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
        .expect("split-material close-wall slot should run through the root solid pipeline");

    let material_region_ids = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    let source_edge_count = mesh
        .boundary_edges
        .iter()
        .flat_map(|edge| edge.provenance.iter())
        .filter(|provenance| provenance.source_entity_kind == SourceEntityKind::Edge)
        .map(|provenance| provenance.source_entity_id.clone())
        .collect::<BTreeSet<_>>()
        .len();

    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "holed_polyhedron"
    );
    assert_eq!(
        material_region_ids,
        BTreeSet::from(["region_base", "region_cap"])
    );
    assert_eq!(mesh.backend.tetrahedron_material_region_count, 2);
    assert_eq!(
        mesh.backend.tetrahedron_unclassified_material_element_count,
        0
    );
    assert_eq!(
        source_edge_count,
        mesh.backend.plc_input_protected_edge_count
    );
    assert_eq!(mesh.backend.tetrahedron_missing_recovery_item_count, 0);
    assert!(
        mesh.backend.tetrahedron_min_exact_scaled_jacobian >= 0.15,
        "min exact scaled Jacobian was {}",
        mesh.backend.tetrahedron_min_exact_scaled_jacobian
    );
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["region_base".to_string(), "region_cap".to_string()],
            require_boundary_source_edge_provenance: true,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("split-material close-wall slot should be solve-ready with source-edge provenance");
}

#[test]
fn generated_solid_mesh_maps_solver_fields_to_boundary_visualization_topology() {
    let geometry = split_material_through_hole_plate_geometry();
    let mesh = generate_analysis_mesh(&geometry, VolumeMeshingOptions::default())
        .expect("split-region through-hole solid should generate a solver mesh");
    let element_values = mesh
        .volume_elements
        .iter()
        .enumerate()
        .map(|(index, _)| index as f64 + 1.0)
        .collect::<Vec<_>>();
    let node_values = mesh
        .nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();

    let boundary_scalars = map_volume_scalar_field_to_boundary_faces(&mesh, &element_values)
        .expect("generated solver element values should map to boundary faces");
    let boundary_node_vectors = map_nodal_vector_field_to_boundary_nodes(&mesh, &node_values)
        .expect("generated solver node values should map to boundary nodes");
    let boundary_face_vectors = map_nodal_vector_field_to_boundary_faces(&mesh, &node_values)
        .expect("generated solver node values should map to boundary face averages");

    assert_eq!(boundary_scalars.len(), mesh.boundary_faces.len());
    assert_eq!(boundary_face_vectors.len(), mesh.boundary_faces.len());
    assert_eq!(
        boundary_node_vectors.len(),
        generated_boundary_node_count(&mesh.boundary_faces)
    );

    let first_face = mesh
        .boundary_faces
        .first()
        .expect("generated mesh should expose boundary faces");
    let expected_scalar = expected_boundary_face_scalar(&mesh.volume_elements, first_face);
    let mapped_scalar = boundary_scalars
        .iter()
        .find(|value| value.face_id == first_face.face_id)
        .expect("first boundary face should have a mapped scalar");
    assert!((mapped_scalar.value - expected_scalar).abs() <= f64::EPSILON);

    let expected_vector = expected_boundary_face_vector(&mesh.nodes, first_face);
    let mapped_vector = boundary_face_vectors
        .iter()
        .find(|value| value.face_id == first_face.face_id)
        .expect("first boundary face should have a mapped vector");
    for component in 0..3 {
        assert!(
            (mapped_vector.value[component] - expected_vector[component]).abs() <= f64::EPSILON
        );
    }
}

fn generated_boundary_node_count(boundary_faces: &[AnalysisBoundaryFace]) -> usize {
    boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<BTreeSet<_>>()
        .len()
}

fn expected_boundary_face_scalar(
    volume_elements: &[AnalysisVolumeElement],
    face: &AnalysisBoundaryFace,
) -> f64 {
    let values_by_element_id = volume_elements
        .iter()
        .enumerate()
        .map(|(index, element)| (element.element_id.as_str(), index as f64 + 1.0))
        .collect::<BTreeMap<_, _>>();
    face.adjacent_volume_element_ids
        .iter()
        .map(|element_id| values_by_element_id[element_id.as_str()])
        .sum::<f64>()
        / face.adjacent_volume_element_ids.len() as f64
}

fn expected_boundary_face_vector(
    nodes: &[AnalysisMeshNode],
    face: &AnalysisBoundaryFace,
) -> [f64; 3] {
    let coordinates_by_node_id = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut value = [0.0_f64; 3];
    for node_id in &face.node_ids {
        let coordinates = coordinates_by_node_id[node_id];
        for component in 0..3 {
            value[component] += coordinates[component];
        }
    }
    for component in &mut value {
        *component /= face.node_ids.len() as f64;
    }
    value
}

#[test]
fn auto_backend_preserves_recovered_material_regions_for_dented_corner_solid() {
    let mesh = generate_analysis_mesh(
        &split_material_dented_corner_box_geometry(),
        VolumeMeshingOptions::default(),
    )
    .expect("split-region dented-corner solid should recover material ownership");

    let material_region_ids = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        material_region_ids,
        BTreeSet::from(["region_base", "region_cap"])
    );
    assert!(mesh
        .volume_elements
        .iter()
        .all(|element| element.material_region_id != "unclassified"));
    assert_eq!(
        mesh.backend
            .tetrahedron_recovered_material_interface_recovery_item_count,
        2
    );
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "star_shaped_polyhedron"
    );
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            required_material_region_ids: vec!["region_base".to_string(), "region_cap".to_string()],
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("dented-corner solid artifact should expose both recovered material regions");
}

#[test]
fn explicit_structured_grid_tetrahedron_backend_runs_structured_stage() {
    let mesh = generate_analysis_mesh(
        &cube_geometry(),
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("explicit structured-grid Tetrahedron backend should run explicitly");

    assert_eq!(mesh.backend.backend, "structured_grid_tetrahedron");
}

#[test]
fn auto_backend_rejects_open_shell_before_volume_meshing() {
    let err = generate_analysis_mesh(&open_shell_cube_geometry(), VolumeMeshingOptions::default())
        .expect_err("open shell should fail before volume meshing");

    assert!(matches!(
        err,
        SolidMeshingError::SourceTopology(SourceTopologyError::OpenBoundaryEdge { count: 1, .. })
    ));
}

#[test]
fn auto_backend_rejects_nonmanifold_edge_before_volume_meshing() {
    let err = generate_analysis_mesh(
        &nonmanifold_edge_cube_geometry(),
        VolumeMeshingOptions::default(),
    )
    .expect_err("nonmanifold edge should fail before volume meshing");

    assert!(matches!(
        err,
        SolidMeshingError::SourceTopology(SourceTopologyError::OpenBoundaryEdge { count: 3, .. })
    ));
}

fn cube_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_cube".to_string(),
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
        regions: vec![Region {
            region_id: "region_boundary".to_string(),
            name: "boundary".to_string(),
            tag: Some("boundary".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_boundary",
            "cube_surface",
            12,
        )],
        diagnostics: Vec::new(),
    }
}

fn open_shell_cube_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_root_meshing_open_shell_cube".to_string();
    geometry.source.path = "/fixtures/generic_open_shell_cube.step".to_string();
    geometry.source.sha256 = "generic-open-shell-cube".to_string();
    geometry.meshes[0].element_count = 10;
    geometry.surface_meshes[0].triangles.truncate(10);
    geometry.region_entity_mappings = vec![RegionEntityMapping::all_faces(
        "region_boundary",
        "cube_surface",
        10,
    )];
    geometry
}

fn nonmanifold_edge_cube_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_root_meshing_nonmanifold_edge_cube".to_string();
    geometry.source.path = "/fixtures/generic_nonmanifold_edge_cube.step".to_string();
    geometry.source.sha256 = "generic-nonmanifold-edge-cube".to_string();
    geometry.meshes[0].element_count = 13;
    geometry.surface_meshes[0].triangles.push([0, 1, 4]);
    geometry.region_entity_mappings = vec![RegionEntityMapping::all_faces(
        "region_boundary",
        "cube_surface",
        13,
    )];
    geometry
}

fn split_material_cube_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_root_meshing_split_material_cube".to_string();
    geometry.regions = vec![
        Region {
            region_id: "region_base".to_string(),
            name: "base".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
        Region {
            region_id: "region_cap".to_string(),
            name: "cap".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
    ];
    geometry.region_entity_mappings = vec![
        RegionEntityMapping::new(
            "region_base",
            "cube_surface",
            EntityKind::Face,
            vec![EntityIdRange::new(0, 6)],
        ),
        RegionEntityMapping::new(
            "region_cap",
            "cube_surface",
            EntityKind::Face,
            vec![EntityIdRange::new(6, 6)],
        ),
    ];
    geometry
}

fn thin_arm_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_root_meshing_thin_arm".to_string();
    geometry.source.path = "/fixtures/generic_thin_arm.step".to_string();
    geometry.source.sha256 = "generic-thin-arm".to_string();
    geometry.meshes[0].mesh_id = "thin_arm_surface".to_string();
    geometry.surface_meshes[0].mesh_id = "thin_arm_surface".to_string();
    for vertex in &mut geometry.surface_meshes[0].vertices {
        vertex[0] *= 3.0;
        vertex[1] *= 0.75;
        vertex[2] *= 0.75;
    }
    geometry.regions = vec![Region {
        region_id: "body".to_string(),
        name: "body".to_string(),
        tag: Some("material".to_string()),
        cad_ownership: None,
    }];
    geometry.region_entity_mappings = vec![RegionEntityMapping::all_faces(
        "body",
        "thin_arm_surface",
        12,
    )];
    geometry
}

fn dented_corner_box_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_root_meshing_dented_corner_box".to_string();
    geometry.source.path = "/fixtures/generic_dented_corner_box.step".to_string();
    geometry.source.sha256 = "generic-dented-corner-box".to_string();
    geometry.surface_meshes[0].vertices[6] = [0.55, 0.55, 0.55];
    geometry
}

fn split_material_dented_corner_box_geometry() -> GeometryAsset {
    let mut geometry = split_material_cube_geometry();
    geometry.geometry_id = "geo_root_meshing_split_material_dented_corner_box".to_string();
    geometry.source.path = "/fixtures/generic_split_material_dented_corner_box.step".to_string();
    geometry.source.sha256 = "generic-split-material-dented-corner-box".to_string();
    geometry.surface_meshes[0].vertices[6] = [0.55, 0.55, 0.55];
    geometry
}

fn close_parallel_wall_slot_geometry() -> GeometryAsset {
    let mut geometry = through_hole_plate_geometry();
    geometry.geometry_id = "geo_root_meshing_close_parallel_wall_slot".to_string();
    geometry.source.path = "/fixtures/generic_close_parallel_wall_slot.step".to_string();
    geometry.source.sha256 = "generic-close-parallel-wall-slot".to_string();
    geometry.meshes[0].mesh_id = "close_parallel_wall_slot_surface".to_string();
    geometry.surface_meshes[0].mesh_id = "close_parallel_wall_slot_surface".to_string();
    for vertex_index in [4_usize, 7, 12, 15] {
        geometry.surface_meshes[0].vertices[vertex_index][0] = 1.4;
    }
    for vertex_index in [5_usize, 6, 13, 14] {
        geometry.surface_meshes[0].vertices[vertex_index][0] = 1.6;
    }
    geometry.region_entity_mappings = vec![RegionEntityMapping::all_faces(
        "body",
        "close_parallel_wall_slot_surface",
        32,
    )];
    geometry
}

fn split_material_close_parallel_wall_slot_geometry() -> GeometryAsset {
    let mut geometry = close_parallel_wall_slot_geometry();
    geometry.geometry_id = "geo_root_meshing_split_material_close_parallel_wall_slot".to_string();
    geometry.source.path =
        "/fixtures/generic_split_material_close_parallel_wall_slot.step".to_string();
    geometry.source.sha256 = "generic-split-material-close-parallel-wall-slot".to_string();
    geometry.regions = vec![
        Region {
            region_id: "region_base".to_string(),
            name: "base".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
        Region {
            region_id: "region_cap".to_string(),
            name: "cap".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
    ];
    geometry.region_entity_mappings = vec![
        RegionEntityMapping::new(
            "region_base",
            "close_parallel_wall_slot_surface",
            EntityKind::Face,
            vec![EntityIdRange::new(16, 4), EntityIdRange::new(24, 4)],
        ),
        RegionEntityMapping::new(
            "region_cap",
            "close_parallel_wall_slot_surface",
            EntityKind::Face,
            vec![EntityIdRange::new(20, 4), EntityIdRange::new(28, 4)],
        ),
    ];
    geometry
}

fn faceted_cylinder_geometry() -> GeometryAsset {
    let segment_count = 8_u32;
    let mut vertices = Vec::<[f64; 3]>::new();
    for z in [0.0_f64, 1.0] {
        for segment_index in 0..segment_count {
            let angle = std::f64::consts::TAU * segment_index as f64 / segment_count as f64;
            vertices.push([angle.cos(), angle.sin(), z]);
        }
    }

    let mut triangles = Vec::<[u32; 3]>::new();
    for segment_index in 1..(segment_count - 1) {
        triangles.push([0, segment_index + 1, segment_index]);
    }
    let top_offset = segment_count;
    for segment_index in 1..(segment_count - 1) {
        triangles.push([
            top_offset,
            top_offset + segment_index,
            top_offset + segment_index + 1,
        ]);
    }
    for segment_index in 0..segment_count {
        let next_segment_index = (segment_index + 1) % segment_count;
        triangles.push([
            segment_index,
            next_segment_index,
            top_offset + next_segment_index,
        ]);
        triangles.push([
            segment_index,
            top_offset + next_segment_index,
            top_offset + segment_index,
        ]);
    }

    let element_count = triangles.len() as u64;
    GeometryAsset {
        geometry_id: "geo_root_meshing_faceted_cylinder".to_string(),
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
            element_count,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "faceted_cylinder_surface",
            vertices,
            triangles,
        )],
        regions: vec![Region {
            region_id: "body".to_string(),
            name: "body".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "body",
            "faceted_cylinder_surface",
            element_count,
        )],
        diagnostics: Vec::new(),
    }
}

fn octahedron_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_octahedron".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_octahedron.step".to_string(),
            sha256: "generic-octahedron".to_string(),
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
            mesh_id: "octahedron_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 6,
            element_count: 8,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "octahedron_surface",
            vec![
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            vec![
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 2, 1],
                [5, 3, 2],
                [5, 4, 3],
                [5, 1, 4],
            ],
        )],
        regions: vec![Region {
            region_id: "region_boundary".to_string(),
            name: "boundary".to_string(),
            tag: Some("boundary".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_boundary",
            "octahedron_surface",
            8,
        )],
        diagnostics: Vec::new(),
    }
}

fn tetrahedron_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_tetrahedron".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_tetrahedron.step".to_string(),
            sha256: "generic-tetrahedron".to_string(),
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
            mesh_id: "tetrahedron_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 4,
            element_count: 4,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "tetrahedron_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )],
        regions: vec![Region {
            region_id: "region_boundary".to_string(),
            name: "boundary".to_string(),
            tag: Some("boundary".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_boundary",
            "tetrahedron_surface",
            4,
        )],
        diagnostics: Vec::new(),
    }
}

fn nested_tetrahedron_shell_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_nested_tetrahedron_shell".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_nested_tetrahedron_shell.step".to_string(),
            sha256: "generic-nested-tetrahedron-shell".to_string(),
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
            mesh_id: "nested_tetrahedron_shell_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 8,
            element_count: 8,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "nested_tetrahedron_shell_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.2, 0.2],
                [0.3, 0.2, 0.2],
                [0.2, 0.3, 0.2],
                [0.2, 0.2, 0.3],
            ],
            vec![
                [0, 2, 1],
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
                [4, 6, 5],
                [4, 5, 7],
                [5, 6, 7],
                [6, 4, 7],
            ],
        )],
        regions: vec![Region {
            region_id: "region_boundary".to_string(),
            name: "boundary".to_string(),
            tag: Some("boundary".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_boundary",
            "nested_tetrahedron_shell_surface",
            8,
        )],
        diagnostics: Vec::new(),
    }
}
