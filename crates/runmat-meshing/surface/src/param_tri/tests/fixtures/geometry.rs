use runmat_geometry_core::{
    CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
    CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, Region,
    RegionEntityMapping,
};

pub(in crate::param_tri::tests) fn geometry_for_topology() -> runmat_geometry_core::GeometryAsset {
    runmat_geometry_core::GeometryAsset {
        geometry_id: "geo".to_string(),
        source: runmat_geometry_core::GeometrySource {
            path: "/fixtures/surface.step".to_string(),
            sha256: "surface".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: runmat_geometry_core::SourceGeometry {
            kind: runmat_geometry_core::SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: runmat_geometry_core::TessellationProfile::default(),
        units: runmat_geometry_core::UnitSystem::Meter,
        revision: 1,
        meshes: Vec::new(),
        surface_meshes: Vec::new(),
        regions: Vec::new(),
        region_entity_mappings: Vec::new(),
        diagnostics: Vec::new(),
    }
}

pub(in crate::param_tri::tests) fn geometry_with_face_domain_sample(
) -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_for_topology();
    geometry.regions = vec![Region {
        region_id: "face_a".to_string(),
        name: "face".to_string(),
        tag: Some("cad_face".to_string()),
        cad_ownership: Some(CadRegionOwnership {
            face_id: Some(7),
            label: Some(CadLabelRef {
                label_entry: "0:1:7".to_string(),
                name: "face".to_string(),
                kind: CadSemanticKind::Face,
            }),
            owner_path: Vec::new(),
            layers: Vec::new(),
            color: None,
            material: None,
        }),
    }];
    geometry.region_entity_mappings = vec![RegionEntityMapping {
        region_id: "face_a".to_string(),
        mesh_id: "surface".to_string(),
        entity_kind: EntityKind::Face,
        ranges: vec![EntityIdRange {
            start: 11,
            count: 1,
        }],
    }];
    geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
        evaluator_id: "cad_evaluator_test".to_string(),
        backend: "test".to_string(),
        format_name: "step".to_string(),
        requires_source_geometry: true,
        faces: vec![CadFaceEvaluator {
            evaluator_id: "cad_face_7".to_string(),
            imported_face_id: 7,
            name: "face".to_string(),
            supports_point_evaluation: true,
            supports_projection: true,
            supports_normal: true,
            supports_derivatives: true,
            supports_curvature: true,
            reference_point_m: Some([0.25, 0.25, 0.0]),
            reference_unit_normal: Some([0.0, 0.0, 1.0]),
            evaluation_samples: vec![
                CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [0.25, 0.25, 0.03],
                    uv: Some([0.25, 0.25]),
                    projected_point_m: Some([0.25, 0.25, 0.0]),
                    unit_normal: Some([0.0, 0.0, 1.0]),
                    projection_error_m: Some(0.03),
                },
                CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [1.25, 0.25, 0.0],
                    uv: Some([1.25, 0.25]),
                    projected_point_m: Some([1.25, 0.25, 0.0]),
                    unit_normal: Some([0.0, 0.0, 1.0]),
                    projection_error_m: Some(0.0),
                },
            ],
        }],
        curves: Vec::new(),
    }];
    geometry
}

pub(in crate::param_tri::tests) fn geometry_with_area_regressing_face_samples(
) -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_with_face_domain_sample();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.30, 0.10, 0.0],
            uv: Some([0.30, 0.10]),
            projected_point_m: Some([0.30, 0.10, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.75, 0.10, 0.0],
            uv: Some([0.75, 0.10]),
            projected_point_m: Some([0.75, 0.10, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.70, 0.25, 0.0],
            uv: Some([0.70, 0.25]),
            projected_point_m: Some([0.70, 0.25, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
    ];
    geometry
}

pub(in crate::param_tri::tests) fn geometry_with_edge_hit_face_samples(
) -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_with_face_domain_sample();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.50, 0.25, 0.0],
            uv: Some([0.50, 0.25]),
            projected_point_m: Some([0.50, 0.25, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.25, 0.125, 0.0],
            uv: Some([0.25, 0.125]),
            projected_point_m: Some([0.25, 0.125, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
    ];
    geometry
}

pub(in crate::param_tri::tests) fn geometry_with_concave_trim_rejected_sample(
) -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_with_face_domain_sample();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
        vec![CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.2, 0.0],
            uv: Some([0.5, 0.2]),
            projected_point_m: Some([0.5, 0.2, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        }];
    geometry
}
