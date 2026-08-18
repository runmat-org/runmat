use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::MetricFieldRequest;
use runmat_meshing_curve::SharedCurveSegmentSplit;

use crate::{
    ExactFaceMetricError, ExactFacePslg, ExactFacePslgSegmentSource, ResolvedFaceMetricField,
};

use super::{
    ExactFaceCandidateDisposition, ExactFaceRefinementCandidate, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind, ExactProtectedSegmentSplit,
};

pub fn classify_exact_face_refinement_candidate(
    candidate: &ExactFaceRefinementCandidate,
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
) -> Result<ExactFaceCandidateDisposition, ExactFaceRefinementError> {
    if candidate.source_face_id != pslg.source_face_id
        || candidate.uv.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            &pslg.source_face_id,
            "candidate face or UV is inconsistent with the PSLG",
        ));
    }
    let field = ResolvedFaceMetricField::new(topology, request)
        .map_err(|error| metric(&pslg.source_face_id, error))?;
    for (segment_index, segment) in pslg.segments.iter().enumerate() {
        control.checkpoint().map_err(|error| {
            ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
                ),
                &pslg.source_face_id,
                error.reason,
            )
        })?;
        let endpoints = segment
            .vertex_indices
            .map(|index| pslg.vertices.get(index as usize));
        let [Some(first), Some(second)] = endpoints else {
            return Err(invalid(
                &pslg.source_face_id,
                "protected segment references an absent PSLG vertex",
            ));
        };
        let midpoint = [
            (first.uv[0] + second.uv[0]) * 0.5,
            (first.uv[1] + second.uv[1]) * 0.5,
        ];
        let metric = field
            .evaluate(&pslg.source_face_id, midpoint, evaluator, control)
            .map_err(|error| metric(&pslg.source_face_id, error))?
            .sizing_metric;
        let left = [candidate.uv[0] - first.uv[0], candidate.uv[1] - first.uv[1]];
        let right = [
            candidate.uv[0] - second.uv[0],
            candidate.uv[1] - second.uv[1],
        ];
        let diametral_product = left[0] * (metric.uu * right[0] + metric.uv * right[1])
            + left[1] * (metric.uv * right[0] + metric.vv * right[1]);
        if !diametral_product.is_finite() {
            return Err(invalid(
                &pslg.source_face_id,
                "protected-segment encroachment measure is not finite",
            ));
        }
        if diametral_product <= 0.0 {
            let ExactFacePslgSegmentSource::ExactTrim {
                source_coedge_id,
                source_edge_id,
            } = &segment.source
            else {
                return Err(invalid(
                    &pslg.source_face_id,
                    "chart-cut refinement requires a face-owned protected split",
                ));
            };
            let Some(segment_parameters) = segment.edge_parameters else {
                return Err(invalid(
                    &pslg.source_face_id,
                    "exact trim segment is missing curve parameters",
                ));
            };
            let (endpoint_node_ids, edge_parameters) =
                if segment_parameters[0] < segment_parameters[1] {
                    ([first.node_id, second.node_id], segment_parameters)
                } else if segment_parameters[1] < segment_parameters[0] {
                    (
                        [second.node_id, first.node_id],
                        [segment_parameters[1], segment_parameters[0]],
                    )
                } else {
                    return Err(invalid(
                        &pslg.source_face_id,
                        "protected segment edge parameters are not distinct",
                    ));
                };
            let split_parameter = edge_parameters[0] * 0.5 + edge_parameters[1] * 0.5;
            if !split_parameter.is_finite()
                || split_parameter == edge_parameters[0]
                || split_parameter == edge_parameters[1]
            {
                return Err(invalid(
                    &pslg.source_face_id,
                    "protected segment has no representable interior split parameter",
                ));
            }
            return Ok(ExactFaceCandidateDisposition::SplitProtectedSegment(
                Box::new(ExactProtectedSegmentSplit {
                    source_face_id: pslg.source_face_id.clone(),
                    pslg_segment_index: segment_index as u32,
                    source_coedge_id: source_coedge_id.clone(),
                    curve_split: SharedCurveSegmentSplit {
                        source_edge_id: source_edge_id.clone(),
                        endpoint_node_ids,
                        edge_parameters,
                        split_parameter,
                    },
                }),
            ));
        }
    }
    Ok(ExactFaceCandidateDisposition::Insert)
}

fn metric(
    face_id: &runmat_geometry_core::PersistentEntityId,
    error: ExactFaceMetricError,
) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Metric(error.kind),
        error.source_face_id.as_ref().unwrap_or(face_id),
        error.reason,
    )
}

fn invalid(
    face_id: &runmat_geometry_core::PersistentEntityId,
    reason: impl Into<String>,
) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::InvalidGeometry,
        face_id,
        reason,
    )
}
