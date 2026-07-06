use super::*;
use runmat_geometry_core::{
    GeometrySource, MaterialEvidence, MeshDescriptor, Region, RegionEntityMapping, SourceGeometry,
    SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
};

fn sample_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_meshing_test".to_string(),
        source: GeometrySource {
            path: "/fixtures/cube.stl".to_string(),
            sha256: "mesh-test".to_string(),
            importer_version: "stl/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Mesh,
            assembly: None,
            material_evidence: vec![MaterialEvidence {
                source_key: "fixture".to_string(),
                normalized_key: "fixture".to_string(),
                value: "steel".to_string(),
                confidence: runmat_geometry_core::MaterialEvidenceConfidence::High,
                unit_basis: None,
                assumptions: Vec::new(),
            }],
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 7,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_a".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 120,
            element_count: 200,
        }],
        surface_meshes: Vec::new(),
        regions: vec![Region {
            region_id: "region_main".to_string(),
            name: "main".to_string(),
            tag: None,
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces("region_main", "mesh_a", 200)],
        diagnostics: Vec::new(),
    }
}

#[test]
fn meshing_prep_is_deterministic() {
    let geometry = sample_geometry();
    let first = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
        .expect("first meshing prep should work");
    let second = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
        .expect("second meshing prep should work");
    assert_eq!(first, second);
    let descriptor = first
        .prepared_meshes
        .first()
        .expect("prepared mesh descriptor should exist");
    assert!(descriptor.region_span_hint >= 1);
}

#[test]
fn meshing_prep_carries_element_geometry_metrics() {
    let mut geometry = sample_geometry();
    geometry.meshes[0].vertex_count = 4;
    geometry.meshes[0].element_count = 2;
    geometry.surface_meshes = vec![SurfaceMesh::new(
        "mesh_a",
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )];

    let prep = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
        .expect("meshing prep should work");
    let descriptor = prep
        .prepared_meshes
        .first()
        .expect("prepared mesh descriptor should exist");
    assert_eq!(descriptor.element_geometry_node_count, 4);
    assert_eq!(descriptor.element_geometry_edge_count, 5);
    assert!(descriptor.mean_element_edge_length_m > 1.0);
    assert!((descriptor.mean_element_area_m2 - 0.5).abs() < 1.0e-12);
    assert_eq!(descriptor.element_geometry_coverage_ratio, 1.0);
    assert_eq!(
        descriptor.reference_element_coordinates_m,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    );
    assert!((descriptor.reference_element_area_m2 - 0.5).abs() < 1.0e-12);
    assert_eq!(descriptor.control_volume_cell_count, 2);
    assert_eq!(descriptor.control_volume_face_count, 5);
    assert_eq!(descriptor.control_volume_internal_face_count, 1);
    assert_eq!(descriptor.control_volume_boundary_face_count, 4);
    assert_eq!(descriptor.control_volume_connectivity_coverage_ratio, 1.0);
    assert_eq!(descriptor.element_topology_sample_element_count, 2);
    assert_eq!(descriptor.element_topology_sample_edge_count, 5);
    assert_eq!(descriptor.element_topology_sample_edge_nodes[0], [0, 1]);
    assert_eq!(
        descriptor.element_topology_sample_node_coordinates_m[0],
        [0.0, 0.0, 0.0]
    );
    assert_eq!(
        descriptor.element_topology_sample_element_edges[0],
        [0, 1, 2]
    );
    assert_eq!(
        descriptor.element_topology_sample_element_orientations[0],
        [1, 1, -1]
    );
    assert!((descriptor.element_topology_sample_element_areas_m2[0] - 0.5).abs() < 1.0e-12);
}

#[test]
fn meshing_prep_validates_options() {
    let geometry = sample_geometry();
    let error = prepare_geometry_for_analysis(
        &geometry,
        MeshingOptions {
            profile: MeshingProfile::AnalysisReady,
            target_element_budget: 0,
        },
    )
    .expect_err("zero budget should fail");
    assert!(error.contains("target_element_budget"));
}

#[test]
fn metadata_only_regions_are_rejected_without_mesh_mapping() {
    let mut geometry = sample_geometry();
    geometry.region_entity_mappings.clear();

    let err = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
        .expect_err("metadata-only region should fail closed");

    assert!(err.contains("region region_main has no mesh entity mapping"));
}

#[test]
fn stale_region_mesh_mappings_are_rejected() {
    let mut geometry = sample_geometry();
    geometry.region_entity_mappings = vec![RegionEntityMapping::all_faces(
        "region_main",
        "missing_mesh",
        200,
    )];

    let err = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
        .expect_err("stale region mapping should fail closed");

    assert!(err.contains("region region_main references unknown mesh entity mapping(s)"));
    assert!(err.contains("missing_mesh"));
}

#[test]
fn adaptive_refine_profile_improves_quality_within_budget() {
    let geometry = sample_geometry();
    let analysis_ready = prepare_geometry_for_analysis(
        &geometry,
        MeshingOptions {
            profile: MeshingProfile::AnalysisReady,
            target_element_budget: 4_000,
        },
    )
    .expect("analysis-ready prep should work");
    let adaptive = prepare_geometry_for_analysis(
        &geometry,
        MeshingOptions {
            profile: MeshingProfile::AdaptiveRefine,
            target_element_budget: 4_000,
        },
    )
    .expect("adaptive-refine prep should work");

    let adaptive_total_elements = adaptive
        .prepared_meshes
        .iter()
        .map(|mesh| mesh.element_count)
        .sum::<u64>();
    assert!(adaptive_total_elements <= 4_000);
    assert!(adaptive.quality.min_scaled_jacobian >= analysis_ready.quality.min_scaled_jacobian);
    assert!(adaptive.quality.mean_aspect_ratio <= analysis_ready.quality.mean_aspect_ratio);
}
