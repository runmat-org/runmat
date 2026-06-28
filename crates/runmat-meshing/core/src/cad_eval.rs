use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    cad_topology::{CadFace, CadTopologyModel},
    predicate::{dot, norm, scale, sub, triangle_centroid, Point3, Triangle3},
    source_topology::SourceTopologyModel,
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadEvaluationSource {
    ParametricCad,
    ImportedEvaluatorSamples,
    PlanarFacetApproximation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadFaceEvaluationFrame {
    pub face_id: String,
    pub source_face_id: u32,
    pub origin_m: Point3,
    pub u_axis: Point3,
    pub v_axis: Point3,
    pub unit_normal: Point3,
    pub area_m2: f64,
    #[serde(default)]
    pub evaluator_backed: bool,
    #[serde(default)]
    pub exact_query_backed: bool,
    #[serde(default)]
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub evaluator_max_projection_error_m: f64,
    #[serde(default)]
    pub evaluator_samples: Vec<CadFaceEvaluationSample>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CadFaceProjection {
    pub point_m: Point3,
    pub uv: [f64; 2],
    pub distance_m: f64,
    pub unit_normal: Point3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEvaluationReport {
    pub source: CadEvaluationSource,
    pub face_frame_count: usize,
    pub evaluator_face_count: usize,
    #[serde(default)]
    pub exact_query_face_count: usize,
    #[serde(default)]
    pub evaluator_sample_count: usize,
    pub normal_query_count: usize,
    pub projection_query_count: usize,
    pub max_projection_error_m: f64,
    pub max_normal_deviation: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEvaluationModel {
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source: CadEvaluationSource,
    pub face_frames: Vec<CadFaceEvaluationFrame>,
    pub report: CadEvaluationReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CadEvaluationError {
    EmptyFaces,
    MissingSourceFace { source_face_id: u32 },
    MissingSourceVertex { vertex_id: u32 },
    DegenerateFace { source_face_id: u32 },
}

impl std::fmt::Display for CadEvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFaces => write!(formatter, "CAD evaluation model has no faces"),
            Self::MissingSourceFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is missing")
            }
            Self::MissingSourceVertex { vertex_id } => {
                write!(formatter, "source vertex {vertex_id} is missing")
            }
            Self::DegenerateFace { source_face_id } => {
                write!(formatter, "source face {source_face_id} is degenerate")
            }
        }
    }
}

impl std::error::Error for CadEvaluationError {}

pub fn build_cad_evaluation_model(
    cad_topology: &CadTopologyModel,
    topology: &SourceTopologyModel,
) -> Result<CadEvaluationModel, CadEvaluationError> {
    if cad_topology.faces.is_empty() {
        return Err(CadEvaluationError::EmptyFaces);
    }
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    let mut frames = Vec::<CadFaceEvaluationFrame>::with_capacity(cad_topology.faces.len());
    for face in &cad_topology.faces {
        let source_face_id = face
            .source_face_ids
            .first()
            .copied()
            .ok_or(CadEvaluationError::MissingSourceFace { source_face_id: 0 })?;
        let source_face = source_faces
            .get(&source_face_id)
            .ok_or(CadEvaluationError::MissingSourceFace { source_face_id })?;
        let points = [
            topology_vertex(topology, source_face.node_ids[0])?,
            topology_vertex(topology, source_face.node_ids[1])?,
            topology_vertex(topology, source_face.node_ids[2])?,
        ];
        let exact_sample = exact_backend_sample(face);
        let evaluator_max_projection_error_m = evaluator_max_projection_error(face);
        let frame = face_frame(
            face.entity_id.id.clone(),
            source_face_id,
            points,
            exact_sample
                .and_then(|sample| sample.unit_normal)
                .or(face.evaluator_unit_normal)
                .unwrap_or(source_face.unit_normal),
            source_face.area_m2,
            exact_sample
                .map(|sample| sample.point_m)
                .or(face.evaluator_reference_point_m),
            face.evaluator_unit_normal.is_some() || !face.evaluator_samples.is_empty(),
            exact_sample.is_some(),
            face.evaluator_samples.len(),
            evaluator_max_projection_error_m,
            bounded_evaluator_samples(face),
        )?;
        frames.push(frame);
    }
    let evaluator_backed_frame_count = frames.iter().filter(|frame| frame.evaluator_backed).count();
    let exact_query_face_count = frames
        .iter()
        .filter(|frame| frame.exact_query_backed)
        .count();
    let evaluator_sample_count = frames
        .iter()
        .map(|frame| frame.evaluator_sample_count)
        .sum();
    let max_evaluator_projection_error_m = frames
        .iter()
        .map(|frame| frame.evaluator_max_projection_error_m)
        .fold(0.0_f64, f64::max);
    let report = CadEvaluationReport {
        source: evaluation_source(
            frames.len(),
            evaluator_backed_frame_count,
            exact_query_face_count,
        ),
        face_frame_count: frames.len(),
        evaluator_face_count: cad_topology.report.evaluator_face_count,
        exact_query_face_count,
        evaluator_sample_count,
        normal_query_count: frames.len(),
        projection_query_count: frames.len(),
        max_projection_error_m: max_evaluator_projection_error_m,
        max_normal_deviation: 0.0,
    };
    Ok(CadEvaluationModel {
        source_geometry_id: cad_topology.source_geometry_id.clone(),
        source_geometry_revision: cad_topology.source_geometry_revision,
        source: report.source,
        face_frames: frames,
        report,
    })
}

pub fn project_to_face(frame: &CadFaceEvaluationFrame, point: Point3) -> CadFaceProjection {
    let relative = sub(point, frame.origin_m);
    let normal_distance = dot(relative, frame.unit_normal);
    let projected = sub(point, scale(frame.unit_normal, normal_distance));
    let projected_relative = sub(projected, frame.origin_m);
    CadFaceProjection {
        point_m: projected,
        uv: [
            dot(projected_relative, frame.u_axis),
            dot(projected_relative, frame.v_axis),
        ],
        distance_m: normal_distance.abs(),
        unit_normal: frame.unit_normal,
    }
}

pub fn summarize_cad_evaluation(
    model: &CadEvaluationModel,
    topology: &SourceTopologyModel,
) -> Result<CadEvaluationReport, CadEvaluationError> {
    let mut projection_query_count = 0_usize;
    let mut max_projection_error_m = 0.0_f64;
    let mut max_normal_deviation = 0.0_f64;
    let source_faces = topology
        .faces
        .iter()
        .map(|face| (face.face_id, face))
        .collect::<BTreeMap<_, _>>();
    for frame in &model.face_frames {
        let source_face = source_faces.get(&frame.source_face_id).ok_or(
            CadEvaluationError::MissingSourceFace {
                source_face_id: frame.source_face_id,
            },
        )?;
        let points = [
            topology_vertex(topology, source_face.node_ids[0])?,
            topology_vertex(topology, source_face.node_ids[1])?,
            topology_vertex(topology, source_face.node_ids[2])?,
        ];
        for point in points {
            let projection = project_to_face(frame, point);
            projection_query_count += 1;
            max_projection_error_m = max_projection_error_m.max(projection.distance_m);
        }
        max_projection_error_m = max_projection_error_m.max(frame.evaluator_max_projection_error_m);
        max_normal_deviation =
            max_normal_deviation.max(1.0 - dot(frame.unit_normal, source_face.unit_normal).abs());
    }
    Ok(CadEvaluationReport {
        source: model.source,
        face_frame_count: model.face_frames.len(),
        evaluator_face_count: model.report.evaluator_face_count,
        exact_query_face_count: model.report.exact_query_face_count,
        evaluator_sample_count: model.report.evaluator_sample_count,
        normal_query_count: model.face_frames.len(),
        projection_query_count,
        max_projection_error_m,
        max_normal_deviation,
    })
}

fn face_frame(
    face_id: String,
    source_face_id: u32,
    points: Triangle3,
    unit_normal: Point3,
    area_m2: f64,
    evaluator_reference_point_m: Option<Point3>,
    evaluator_backed: bool,
    exact_query_backed: bool,
    evaluator_sample_count: usize,
    evaluator_max_projection_error_m: f64,
    evaluator_samples: Vec<CadFaceEvaluationSample>,
) -> Result<CadFaceEvaluationFrame, CadEvaluationError> {
    let edge = sub(points[1], points[0]);
    let edge_length = norm(edge);
    if edge_length <= f64::EPSILON || norm(unit_normal) <= f64::EPSILON {
        return Err(CadEvaluationError::DegenerateFace { source_face_id });
    }
    let u_axis = scale(edge, 1.0 / edge_length);
    let v_axis = crate::predicate::cross(unit_normal, u_axis);
    let v_length = norm(v_axis);
    if v_length <= f64::EPSILON {
        return Err(CadEvaluationError::DegenerateFace { source_face_id });
    }
    Ok(CadFaceEvaluationFrame {
        face_id,
        source_face_id,
        origin_m: evaluator_reference_point_m.unwrap_or_else(|| triangle_centroid(points)),
        u_axis,
        v_axis: scale(v_axis, 1.0 / v_length),
        unit_normal,
        area_m2,
        evaluator_backed,
        exact_query_backed,
        evaluator_sample_count,
        evaluator_max_projection_error_m,
        evaluator_samples,
    })
}

fn evaluation_source(
    face_frame_count: usize,
    evaluator_backed_frame_count: usize,
    exact_query_face_count: usize,
) -> CadEvaluationSource {
    if face_frame_count > 0 && exact_query_face_count == face_frame_count {
        CadEvaluationSource::ParametricCad
    } else if evaluator_backed_frame_count > 0 {
        CadEvaluationSource::ImportedEvaluatorSamples
    } else {
        CadEvaluationSource::PlanarFacetApproximation
    }
}

fn exact_backend_sample(face: &CadFace) -> Option<&CadFaceEvaluationSample> {
    face.evaluator_samples.iter().find(|sample| {
        sample.source == CadFaceEvaluationSampleSource::BackendQuery
            && finite_point(sample.point_m)
            && sample
                .unit_normal
                .is_some_and(|normal| finite_point(normal) && norm(normal) > 0.0)
            && sample
                .projection_error_m
                .is_none_or(|error| error.is_finite() && error >= 0.0)
    })
}

fn evaluator_max_projection_error(face: &CadFace) -> f64 {
    face.evaluator_samples
        .iter()
        .filter_map(|sample| sample.projection_error_m)
        .filter(|error| error.is_finite() && *error >= 0.0)
        .fold(0.0_f64, f64::max)
}

fn bounded_evaluator_samples(face: &CadFace) -> Vec<CadFaceEvaluationSample> {
    face.evaluator_samples
        .iter()
        .filter(|sample| {
            finite_point(sample.point_m)
                && sample
                    .projected_point_m
                    .is_none_or(|point| finite_point(point))
                && sample
                    .uv
                    .is_none_or(|uv| uv.iter().all(|value| value.is_finite()))
                && sample
                    .unit_normal
                    .is_none_or(|normal| finite_point(normal) && norm(normal) > 0.0)
                && sample
                    .projection_error_m
                    .is_none_or(|error| error.is_finite() && error >= 0.0)
        })
        .take(8)
        .cloned()
        .collect()
}

fn finite_point(point: Point3) -> bool {
    point.iter().all(|coordinate| coordinate.is_finite())
}

fn topology_vertex(
    topology: &SourceTopologyModel,
    vertex_id: u32,
) -> Result<Point3, CadEvaluationError> {
    topology
        .vertices
        .get(vertex_id as usize)
        .filter(|vertex| vertex.vertex_id == vertex_id)
        .map(|vertex| vertex.coordinates_m)
        .ok_or(CadEvaluationError::MissingSourceVertex { vertex_id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_cad_topology, source_topology_from_boundary_input, BoundaryMeshInput};
    use runmat_geometry_core::{
        CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
        CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, Region,
        RegionEntityMapping,
    };

    #[test]
    fn builds_planar_face_evaluation_frames() {
        let topology = cube_topology();
        let geometry = geometry_for_topology();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let report = summarize_cad_evaluation(&model, &topology).expect("summary");

        assert_eq!(model.face_frames.len(), topology.faces.len());
        assert_eq!(report.face_frame_count, topology.faces.len());
        assert_eq!(report.source, CadEvaluationSource::PlanarFacetApproximation);
        assert_eq!(report.evaluator_face_count, 0);
        assert_eq!(report.projection_query_count, topology.faces.len() * 3);
        assert_eq!(report.max_projection_error_m, 0.0);
        assert_eq!(report.max_normal_deviation, 0.0);
    }

    #[test]
    fn projects_points_to_face_frame() {
        let topology = cube_topology();
        let geometry = geometry_for_topology();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");

        let projection = project_to_face(&model.face_frames[0], [0.25, 0.25, 0.5]);

        assert!(projection.distance_m > 0.0);
        assert!(dot(projection.unit_normal, model.face_frames[0].unit_normal) > 0.999);
    }

    #[test]
    fn uses_imported_evaluator_face_samples_when_available() {
        let topology = cube_topology();
        let geometry = geometry_with_face_evaluator();
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");

        assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
        assert_eq!(model.report.evaluator_face_count, 2);
        assert_eq!(model.report.exact_query_face_count, 0);
        assert_eq!(model.report.evaluator_sample_count, 0);
        assert!(model.face_frames.iter().any(|frame| frame.evaluator_backed
            && frame.origin_m == [0.25, 0.25, 0.75]
            && frame.unit_normal == [0.0, 0.0, 1.0]));
    }

    #[test]
    fn exact_backend_query_samples_drive_parametric_cad_frames() {
        let topology = cube_topology();
        let mut geometry = geometry_with_face_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.5, 1.0],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(2.0e-6),
            }];
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

        let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
        let report = summarize_cad_evaluation(&model, &topology).expect("summary");

        assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
        assert_eq!(model.report.evaluator_sample_count, 2);
        assert_eq!(model.report.exact_query_face_count, 2);
        assert_eq!(model.report.max_projection_error_m, 2.0e-6);
        assert_eq!(report.max_projection_error_m, 2.0e-6);
        let frame = model
            .face_frames
            .iter()
            .find(|frame| frame.exact_query_backed)
            .expect("one frame should be exact-query backed");
        assert_eq!(frame.origin_m, [0.5, 0.5, 1.0]);
        assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
        assert_eq!(frame.evaluator_max_projection_error_m, 2.0e-6);
        assert_eq!(frame.evaluator_samples.len(), 1);
        assert_eq!(frame.evaluator_samples[0].uv, Some([0.5, 0.5]));
    }

    fn cube_topology() -> SourceTopologyModel {
        source_topology_from_boundary_input(&BoundaryMeshInput {
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
                crate::BoundaryMeshTriangle {
                    triangle_id: 0,
                    node_ids: [0, 2, 1],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
                crate::BoundaryMeshTriangle {
                    triangle_id: 1,
                    node_ids: [0, 3, 2],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
                crate::BoundaryMeshTriangle {
                    triangle_id: 2,
                    node_ids: [4, 5, 6],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
                crate::BoundaryMeshTriangle {
                    triangle_id: 3,
                    node_ids: [4, 6, 7],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
            ],
            bounds_min_m: [0.0, 0.0, 0.0],
            bounds_max_m: [1.0, 1.0, 1.0],
            region_ids: Vec::new(),
        })
    }

    fn geometry_for_topology() -> runmat_geometry_core::GeometryAsset {
        runmat_geometry_core::GeometryAsset {
            geometry_id: "geo_eval_cube".to_string(),
            source: runmat_geometry_core::GeometrySource {
                path: "/fixtures/eval.step".to_string(),
                sha256: "eval".to_string(),
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

    fn geometry_with_face_evaluator() -> runmat_geometry_core::GeometryAsset {
        let mut geometry = geometry_for_topology();
        geometry.regions = vec![Region {
            region_id: "face_000001".to_string(),
            name: "face".to_string(),
            tag: Some("cad_face".to_string()),
            cad_ownership: Some(CadRegionOwnership {
                face_id: Some(1),
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
}
