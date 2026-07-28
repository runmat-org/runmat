use std::collections::BTreeMap;

use crate::{
    math::{dot, Point3},
    topology::{CadFace, CadTopologyModel, SourceTopologyModel},
};

use super::{
    projection::project_to_face,
    samples::max_optional_curvature,
    types::{
        CadEvaluationError, CadEvaluationModel, CadEvaluationReport, CadEvaluationSource,
        CadFaceEvaluationFrame,
    },
};

pub(super) fn build_cad_evaluation_report(
    cad_topology: &CadTopologyModel,
    frames: &[CadFaceEvaluationFrame],
) -> CadEvaluationReport {
    let evaluator_backed_frame_count = frames.iter().filter(|frame| frame.evaluator_backed).count();
    let live_query_face_count = frames
        .iter()
        .filter(|frame| frame.live_query_backed)
        .count();
    let exact_query_face_count = frames
        .iter()
        .filter(|frame| frame.exact_query_backed)
        .count();
    let point_evaluation_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_point_evaluation
        });
    let projection_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_projection
        });
    let normal_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| face.evaluator_supports_normal);
    let derivative_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_derivatives
        });
    let curvature_supported_face_count =
        evaluator_supported_source_face_count(cad_topology, |face| {
            face.evaluator_supports_curvature
        });
    let derivative_query_count = frames
        .iter()
        .filter(|frame| {
            frame.u_derivative_m_per_uv.is_some() && frame.v_derivative_m_per_uv.is_some()
        })
        .count();
    let curvature_query_count = frames
        .iter()
        .filter(|frame| frame.max_curvature_estimate_1_per_m.is_some())
        .count();

    CadEvaluationReport {
        source: evaluation_source(
            frames.len(),
            evaluator_backed_frame_count,
            live_query_face_count,
            exact_query_face_count,
        ),
        face_frame_count: frames.len(),
        evaluator_face_count: cad_topology.report.evaluator_face_count,
        live_query_face_count,
        exact_query_face_count,
        point_evaluation_supported_face_count,
        projection_supported_face_count,
        normal_supported_face_count,
        derivative_supported_face_count,
        curvature_supported_face_count,
        missing_exact_query_face_count: evaluator_backed_frame_count
            .saturating_sub(exact_query_face_count),
        missing_derivative_query_face_count: derivative_supported_face_count
            .saturating_sub(derivative_query_count),
        missing_curvature_query_face_count: curvature_supported_face_count
            .saturating_sub(curvature_query_count),
        evaluator_sample_count: frames
            .iter()
            .map(|frame| frame.evaluator_sample_count)
            .sum(),
        evaluator_rejected_sample_count: frames
            .iter()
            .map(|frame| frame.evaluator_rejected_sample_count)
            .sum(),
        normal_query_count: frames.len(),
        projection_query_count: frames.len(),
        derivative_query_count,
        curvature_query_count,
        max_projection_error_m: frames
            .iter()
            .map(|frame| frame.evaluator_max_projection_error_m)
            .fold(0.0_f64, f64::max),
        max_normal_deviation: 0.0,
        uv_domain_face_count: frames
            .iter()
            .filter(|frame| frame.uv_bounds.is_some())
            .count(),
        uv_projection_out_of_bounds_count: 0,
        max_curvature_estimate_1_per_m: max_optional_curvature(
            frames
                .iter()
                .filter_map(|frame| frame.max_curvature_estimate_1_per_m),
        ),
    }
}

pub fn summarize_cad_evaluation(
    model: &CadEvaluationModel,
    topology: &SourceTopologyModel,
) -> Result<CadEvaluationReport, CadEvaluationError> {
    let mut projection_query_count = 0_usize;
    let mut max_projection_error_m = 0.0_f64;
    let mut max_normal_deviation = 0.0_f64;
    let mut uv_domain_face_count = 0_usize;
    let mut uv_projection_out_of_bounds_count = 0_usize;
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
            if !projection.uv_in_bounds {
                uv_projection_out_of_bounds_count += 1;
            }
        }
        if frame.uv_bounds.is_some() {
            uv_domain_face_count += 1;
        }
        max_projection_error_m = max_projection_error_m.max(frame.evaluator_max_projection_error_m);
        max_normal_deviation =
            max_normal_deviation.max(1.0 - dot(frame.unit_normal, source_face.unit_normal).abs());
    }
    Ok(CadEvaluationReport {
        source: model.source,
        face_frame_count: model.face_frames.len(),
        evaluator_face_count: model.report.evaluator_face_count,
        live_query_face_count: model.report.live_query_face_count,
        exact_query_face_count: model.report.exact_query_face_count,
        point_evaluation_supported_face_count: model.report.point_evaluation_supported_face_count,
        projection_supported_face_count: model.report.projection_supported_face_count,
        normal_supported_face_count: model.report.normal_supported_face_count,
        derivative_supported_face_count: model.report.derivative_supported_face_count,
        curvature_supported_face_count: model.report.curvature_supported_face_count,
        missing_exact_query_face_count: model.report.missing_exact_query_face_count,
        missing_derivative_query_face_count: model.report.missing_derivative_query_face_count,
        missing_curvature_query_face_count: model.report.missing_curvature_query_face_count,
        evaluator_sample_count: model.report.evaluator_sample_count,
        evaluator_rejected_sample_count: model.report.evaluator_rejected_sample_count,
        normal_query_count: model.face_frames.len(),
        projection_query_count,
        derivative_query_count: model.report.derivative_query_count,
        curvature_query_count: model.report.curvature_query_count,
        max_projection_error_m,
        max_normal_deviation,
        uv_domain_face_count,
        uv_projection_out_of_bounds_count,
        max_curvature_estimate_1_per_m: model.report.max_curvature_estimate_1_per_m,
    })
}

fn evaluation_source(
    _face_frame_count: usize,
    evaluator_backed_frame_count: usize,
    live_query_face_count: usize,
    exact_query_face_count: usize,
) -> CadEvaluationSource {
    if live_query_face_count > 0 {
        CadEvaluationSource::ParametricCad
    } else if evaluator_backed_frame_count > 0 || exact_query_face_count > 0 {
        CadEvaluationSource::ImportedEvaluatorSamples
    } else {
        CadEvaluationSource::PlanarFacetApproximation
    }
}

fn evaluator_supported_source_face_count(
    cad_topology: &CadTopologyModel,
    predicate: impl Fn(&CadFace) -> bool,
) -> usize {
    cad_topology
        .faces
        .iter()
        .filter(|face| predicate(face))
        .map(|face| face.source_face_ids.len().max(1))
        .sum()
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
