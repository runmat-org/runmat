use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    cad_topology::CadTopologyModel,
    predicate::{dot, norm, scale, sub, triangle_centroid, Point3, Triangle3},
    source_topology::SourceTopologyModel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadEvaluationSource {
    ParametricCad,
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
        let frame = face_frame(
            face.entity_id.id.clone(),
            source_face_id,
            points,
            source_face.unit_normal,
            source_face.area_m2,
        )?;
        frames.push(frame);
    }
    let report = CadEvaluationReport {
        source: evaluation_source(cad_topology),
        face_frame_count: frames.len(),
        evaluator_face_count: cad_topology.report.evaluator_face_count,
        normal_query_count: frames.len(),
        projection_query_count: frames.len(),
        max_projection_error_m: 0.0,
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
        max_normal_deviation =
            max_normal_deviation.max(1.0 - dot(frame.unit_normal, source_face.unit_normal).abs());
    }
    Ok(CadEvaluationReport {
        source: model.source,
        face_frame_count: model.face_frames.len(),
        evaluator_face_count: model.report.evaluator_face_count,
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
        origin_m: triangle_centroid(points),
        u_axis,
        v_axis: scale(v_axis, 1.0 / v_length),
        unit_normal,
        area_m2,
    })
}

fn evaluation_source(_cad_topology: &CadTopologyModel) -> CadEvaluationSource {
    CadEvaluationSource::PlanarFacetApproximation
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
}
