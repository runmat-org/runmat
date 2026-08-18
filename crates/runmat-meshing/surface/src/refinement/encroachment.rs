use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::MetricFieldRequest;
use runmat_meshing_curve::SharedCurveSegmentSplit;

use crate::{
    exact_face_chart_cut_node_id, ExactFaceMetricError, ExactFacePslg, ExactFacePslgSegmentSource,
    ResolvedFaceMetricField,
};

use super::{
    ExactChartCutSplit, ExactChartCutSplitImage, ExactFaceCandidateDisposition,
    ExactFaceRefinementCandidate, ExactFaceRefinementError, ExactFaceRefinementErrorKind,
    ExactProtectedSegmentSplit,
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
            let (source_coedge_id, source_edge_id) = match &segment.source {
                ExactFacePslgSegmentSource::ExactTrim {
                    source_coedge_id,
                    source_edge_id,
                } => (source_coedge_id, source_edge_id),
                ExactFacePslgSegmentSource::ChartCut { cut_id } => {
                    return chart_cut_split(pslg, segment_index, *cut_id);
                }
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

fn chart_cut_split(
    pslg: &ExactFacePslg,
    segment_index: usize,
    cut_id: runmat_meshing_core::StableDigest,
) -> Result<ExactFaceCandidateDisposition, ExactFaceRefinementError> {
    let segment = &pslg.segments[segment_index];
    if segment.edge_parameters.is_some() {
        return Err(invalid(
            &pslg.source_face_id,
            "chart cut cannot carry exact curve parameters",
        ));
    }
    let endpoint_node_ids = segment
        .vertex_indices
        .map(|index| pslg.vertices[index as usize].node_id);
    if endpoint_node_ids[0] == endpoint_node_ids[1] {
        return Err(invalid(
            &pslg.source_face_id,
            "chart cut endpoints must have distinct 3D identities",
        ));
    }
    let mut counterparts = pslg
        .segments
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            *index != segment_index
                && candidate.source == ExactFacePslgSegmentSource::ChartCut { cut_id }
                && candidate.edge_parameters.is_none()
                && candidate
                    .vertex_indices
                    .map(|vertex| pslg.vertices[vertex as usize].node_id)
                    == [endpoint_node_ids[1], endpoint_node_ids[0]]
        });
    let Some((counterpart_index, counterpart)) = counterparts.next() else {
        return Err(invalid(
            &pslg.source_face_id,
            "chart cut has no reversed periodic image",
        ));
    };
    if counterparts.next().is_some() {
        return Err(invalid(
            &pslg.source_face_id,
            "chart cut has ambiguous reversed periodic images",
        ));
    }
    let image = |index: usize, vertices: [u32; 2]| ExactChartCutSplitImage {
        pslg_segment_index: index as u32,
        vertex_indices: vertices,
        midpoint_uv: midpoint(vertices.map(|vertex| pslg.vertices[vertex as usize].uv)),
    };
    let mut images = [
        image(segment_index, segment.vertex_indices),
        image(counterpart_index, counterpart.vertex_indices),
    ];
    images.sort_by_key(|image| image.pslg_segment_index);
    Ok(ExactFaceCandidateDisposition::SplitChartCut(Box::new(
        ExactChartCutSplit {
            source_face_id: pslg.source_face_id.clone(),
            cut_id,
            node_id: exact_face_chart_cut_node_id(cut_id, endpoint_node_ids),
            images,
        },
    )))
}

fn midpoint(endpoints: [[f64; 2]; 2]) -> [f64; 2] {
    [
        endpoints[0][0] * 0.5 + endpoints[1][0] * 0.5,
        endpoints[0][1] * 0.5 + endpoints[1][1] * 0.5,
    ]
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
