use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::MetricFieldRequest;
use runmat_meshing_curve::SharedCurveSegmentSplit;
use std::collections::BTreeSet;

use crate::{
    exact_face_chart_cut_node_id, ExactFaceChartParameterization, ExactFaceMetricError,
    ExactFacePslg, ExactFacePslgSegmentSource, ResolvedFaceMetricField,
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
}

pub fn classify_exact_face_refinement_candidate(
    candidate: &ExactFaceRefinementCandidate,
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
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
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
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
    } else {
        Err(invalid(
            &pslg.source_face_id,
            "candidate classification is empty",
        ))
    }
}

pub(crate) fn classify_exact_face_refinement_candidates_in_parameterization(
    candidates: &[ExactFaceRefinementCandidate],
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    parameterization: &ExactFaceChartParameterization,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
) -> Result<ExactFaceCandidateClassifications, ExactFaceRefinementError> {
    if candidates.iter().any(|candidate| {
        candidate.source_face_id != pslg.source_face_id
            || candidate.uv.iter().any(|value| !value.is_finite())
    }) {
        return Err(invalid(
            &pslg.source_face_id,
            "candidate face or UV is inconsistent with the PSLG",
        ));
    }
    let field = ResolvedFaceMetricField::new(topology, request)
        .map_err(|error| metric(&pslg.source_face_id, error))?;
    let mut segment_metrics = Vec::with_capacity(pslg.segments.len());
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
            .evaluate_parameterized(
                &pslg.source_face_id,
                midpoint,
                parameterization,
                evaluator,
                control,
            )
            .map_err(|error| metric(&pslg.source_face_id, error))?
            .sizing_metric;
        segment_metrics.push((segment_index, first, second, metric));
    }
    let mut encroached_segments = BTreeSet::new();
    let mut insertions = Vec::new();
    for candidate in candidates {
        let mut encroached = false;
        for (segment_index, first, second, metric) in &segment_metrics {
            let segment = &pslg.segments[*segment_index];
            if matches!(
                candidate.reason,
                super::ExactFaceRefinementReason::MetricAngle
                    | super::ExactFaceRefinementReason::PhysicalAspectRatio
            ) && matches!(segment.source, ExactFacePslgSegmentSource::ExactTrim { .. })
            {
                // Shared exact curves already satisfy their independently validated metric-size
                // contract. Face-owned shape refinement may split chart cuts, but it must not
                // repeatedly rewrite a global curve merely to improve a local triangle angle or
                // aspect ratio. Geometric-deviation and metric-size candidates remain hard and
                // can request an exact-curve split.
                continue;
            }
            let Some(encroached_by_candidate) =
                metric_diametral_disk_contains(candidate.uv, first.uv, second.uv, *metric)
            else {
                return Err(invalid(
                    &pslg.source_face_id,
                    "protected-segment encroachment measure is not finite",
                ));
            };
            if encroached_by_candidate {
                encroached_segments.insert(*segment_index);
                encroached = true;
            }
        }
        if !encroached {
            insertions.push(candidate.clone());
        }
    }
    let mut protected_splits = Vec::new();
    let mut chart_cut_splits = Vec::new();
    for segment_index in encroached_segments {
        let (_, first, second, _) = &segment_metrics[segment_index];
        match split_protected_segment(pslg, segment_index, first, second)? {
            ExactFaceCandidateDisposition::SplitProtectedSegment(split) => {
                let split = *split;
                if protected_splits
                    .iter()
                    .all(|existing: &ExactProtectedSegmentSplit| {
                        existing.curve_split != split.curve_split
                    })
                {
                    protected_splits.push(split);
                }
            }
            ExactFaceCandidateDisposition::SplitChartCut(split) => {
                let split = *split;
                if !chart_cut_splits.contains(&split) {
                    chart_cut_splits.push(split);
                }
            }
            ExactFaceCandidateDisposition::Insert => {
                return Err(invalid(
                    &pslg.source_face_id,
                    "encroached segment classification did not request a split",
                ));
            }
        }
    }
    Ok(ExactFaceCandidateClassifications {
        protected_splits,
        chart_cut_splits,
        insertions,
    })
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
