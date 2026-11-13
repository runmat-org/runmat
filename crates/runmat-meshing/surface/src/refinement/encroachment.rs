mod classification;
mod influence;
pub(crate) use classification::{
    classify_exact_face_refinement_candidates_in_parameterization, classify_with_context,
};
use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::MetricFieldRequest;
use runmat_meshing_curve::SharedCurveSegmentSplit;

use crate::{
    exact_face_chart_cut_node_id, ExactFaceChartParameterization, ExactFaceMetricError,
    ExactFacePslg, ExactFacePslgSegmentSource,
};

use super::{
    ExactChartCutSplit, ExactChartCutSplitImage, ExactFaceCandidateDisposition,
    ExactFaceRefinementCandidate, ExactFaceRefinementError, ExactFaceRefinementErrorKind,
    ExactProtectedSegmentSplit,
};

pub(crate) struct ExactFaceCandidateClassifications {
    pub protected_splits: Vec<ExactProtectedSegmentSplit>,
    pub chart_cut_splits: Vec<ExactChartCutSplit>,
    pub insertions: Vec<ExactFaceRefinementCandidate>,
    /// Shape-only candidates lying on a shared exact curve; see `ExactFaceCandidateDisposition`.
    pub withdrawn: usize,
}

pub fn classify_exact_face_refinement_candidate(
    candidate: &ExactFaceRefinementCandidate,
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &dyn ExactSurfaceEvaluator,
    control: &dyn GeometryEvaluationControl,
) -> Result<ExactFaceCandidateDisposition, ExactFaceRefinementError> {
    classify_exact_face_refinement_candidate_in_parameterization(
        candidate,
        pslg,
        topology,
        request,
        &ExactFaceChartParameterization::EvaluatorParameters,
        evaluator,
        control,
    )
}

pub(crate) fn classify_exact_face_refinement_candidate_in_parameterization(
    candidate: &ExactFaceRefinementCandidate,
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    parameterization: &ExactFaceChartParameterization,
    evaluator: &dyn ExactSurfaceEvaluator,
    control: &dyn GeometryEvaluationControl,
) -> Result<ExactFaceCandidateDisposition, ExactFaceRefinementError> {
    let mut classifications = classify_exact_face_refinement_candidates_in_parameterization(
        std::slice::from_ref(candidate),
        pslg,
        topology,
        request,
        parameterization,
        evaluator,
        control,
    )?;
    if let Some(split) = classifications.protected_splits.pop() {
        Ok(ExactFaceCandidateDisposition::SplitProtectedSegment(
            Box::new(split),
        ))
    } else if let Some(split) = classifications.chart_cut_splits.pop() {
        Ok(ExactFaceCandidateDisposition::SplitChartCut(Box::new(
            split,
        )))
    } else if classifications.insertions.len() == 1 {
        Ok(ExactFaceCandidateDisposition::Insert)
    } else if classifications.withdrawn == 1 {
        Ok(ExactFaceCandidateDisposition::Withdraw)
    } else {
        Err(invalid(
            &pslg.source_face_id,
            "candidate classification is empty",
        ))
    }
}

/// Conservatively classifies a point against the closed diametral disk in the
/// local metric. The error bound covers the four products and their summation;
/// values indistinguishable from the boundary are treated as encroaching so
/// equivalent CAD parameterizations cannot choose opposite refinement paths
/// from roundoff alone.
pub(super) fn metric_diametral_disk_contains(
    candidate: [f64; 2],
    first: [f64; 2],
    second: [f64; 2],
    metric: crate::ParametricMetricTensor,
) -> Option<bool> {
    let left = [candidate[0] - first[0], candidate[1] - first[1]];
    let right = [candidate[0] - second[0], candidate[1] - second[1]];
    let terms = [
        left[0] * metric.uu * right[0],
        left[0] * metric.uv * right[1],
        left[1] * metric.uv * right[0],
        left[1] * metric.vv * right[1],
    ];
    if terms.iter().any(|term| !term.is_finite()) {
        return None;
    }
    let product = terms.into_iter().sum::<f64>();
    let magnitude = terms.into_iter().map(f64::abs).sum::<f64>();
    let roundoff_bound = 32.0 * f64::EPSILON * magnitude;
    (product.is_finite() && roundoff_bound.is_finite()).then_some(product <= roundoff_bound)
}

fn split_protected_segment(
    pslg: &ExactFacePslg,
    segment_index: usize,
    first: &crate::ExactFacePslgVertex,
    second: &crate::ExactFacePslgVertex,
) -> Result<ExactFaceCandidateDisposition, ExactFaceRefinementError> {
    let segment = &pslg.segments[segment_index];
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
    let (endpoint_node_ids, edge_parameters) = if segment_parameters[0] < segment_parameters[1] {
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
    Ok(ExactFaceCandidateDisposition::SplitProtectedSegment(
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
    ))
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
