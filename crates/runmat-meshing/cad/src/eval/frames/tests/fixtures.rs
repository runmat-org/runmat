use crate::topology::{source_mesh, SourceTopologyModel};
use runmat_geometry_core::{
    CadEvaluatorSet, CadFaceEvaluator, CadLabelRef, CadRegionOwnership, CadSemanticKind,
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, Region, RegionEntityMapping,
    SourceGeometry, SourceGeometryKind, TessellationProfile, UnitSystem,
};

pub(super) fn cube_topology() -> SourceTopologyModel {
    source_mesh::source_topology_from_boundary_input(&source_mesh::SourceTopologyInput {
        mesh_id: "cube_surface".to_string(),
        source_geometry_id: "geo_eval_cube".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        triangles: vec![
            source_mesh::SourceTopologyTriangle {
                triangle_id: 0,
                node_ids: [0, 2, 1],
                region_ids: Vec::new(),
            },
            source_mesh::SourceTopologyTriangle {
                triangle_id: 1,
                node_ids: [0, 3, 2],
                region_ids: Vec::new(),
            },
            source_mesh::SourceTopologyTriangle {
                triangle_id: 2,
                node_ids: [4, 5, 6],
                region_ids: Vec::new(),
            },
            source_mesh::SourceTopologyTriangle {
                triangle_id: 3,
                node_ids: [4, 6, 7],
                region_ids: Vec::new(),
            },
        ],
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [1.0, 1.0, 1.0],
        region_ids: Vec::new(),
    })
}

pub(super) fn geometry_for_topology() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_eval_cube".to_string(),
        source: GeometrySource {
            path: "/fixtures/eval.step".to_string(),
            sha256: "eval".to_string(),
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
        meshes: Vec::new(),
        surface_meshes: Vec::new(),
        regions: Vec::new(),
        region_entity_mappings: Vec::new(),
        diagnostics: Vec::new(),
    }
}

pub(super) fn geometry_with_face_evaluator() -> GeometryAsset {
    let mut geometry = geometry_for_topology();
    geometry.regions = vec![Region {
        region_id: "face_000001".to_string(),
        name: "face".to_string(),
        tag: Some("cad_face".to_string()),
        cad_ownership: Some(CadRegionOwnership {
            face_id: Some(1),
            curve_id: None,
            label: Some(CadLabelRef {
                label_entry: "0:1:1".to_string(),
                name: "face".to_string(),
                kind: CadSemanticKind::Face,
            }),
            owner_path: Vec::new(),
            layers: Vec::new(),
            color: None,
            material: None,
        }),
    }];
    geometry.region_entity_mappings = vec![RegionEntityMapping::new(
        "face_000001",
        "mesh_1",
        EntityKind::Face,
        vec![EntityIdRange::new(2, 2)],
    )];
    geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
        evaluator_id: "cad_evaluator_test".to_string(),
        backend: "test".to_string(),
        format_name: "step".to_string(),
        requires_source_geometry: true,
        faces: vec![CadFaceEvaluator {
            evaluator_id: "cad_face_1".to_string(),
            imported_face_id: 1,
            name: "face".to_string(),
            supports_point_evaluation: true,
            supports_projection: true,
            supports_normal: true,
            supports_derivatives: true,
            supports_curvature: true,
            reference_point_m: Some([0.25, 0.25, 0.75]),
            reference_unit_normal: Some([0.0, 0.0, 1.0]),
            evaluation_samples: Vec::new(),
        }],
        curves: Vec::new(),
    }];
    geometry
}
