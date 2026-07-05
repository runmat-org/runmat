use super::*;
use std::collections::BTreeSet;

use runmat_geometry_core::{
    CadCurveEvaluationSample, CadCurveEvaluationSampleSource, CadCurveEvaluator, CadEvaluatorSet,
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};
use runmat_meshing_core::{
    validate_analysis_mesh, validate_analysis_mesh_with_options, AnalysisFieldTopologyLocation,
    AnalysisMeshValidationOptions, MeshBackendKind, MeshSizingField, MeshTargetSize,
    QualityThresholds, SizingSample, SourceEntityKind, VolumeMeshingOptions,
    ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID, ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
    TETRAHEDRON4_FIELD_ELEMENT_KIND, TRI3_FIELD_ELEMENT_KIND,
};

#[test]
fn auto_backend_recovers_plc_constraints_for_cube() {
    let mesh = generate_analysis_mesh(&cube_geometry(), VolumeMeshingOptions::default())
        .expect("auto backend should run the root solid pipeline");

    assert_eq!(mesh.backend.backend, "solid");
    assert_eq!(mesh.backend.algorithm, "plc_tetrahedron/v1");
    assert_eq!(
        mesh.backend.tetrahedron_generation_family,
        "boundary_conforming_box"
    );
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
fn explicit_sizing_generates_solve_ready_single_tetrahedron_mesh() {
    let mesh = generate_analysis_mesh(
        &tetrahedron_geometry(),
        VolumeMeshingOptions {
            target_size: MeshTargetSize::LengthM(10.0),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("tetrahedron PLC should run through the root solid pipeline");

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
        5
    );
    assert_eq!(mesh.backend.tetrahedron_generation_rejected_family_count, 4);
    assert_eq!(mesh.backend.tetrahedron_generation_selected_family_index, 5);
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
