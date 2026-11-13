//! Classify all protected-segment encroachments with cooperative pairwise walks.
use super::{
    invalid, metric, metric_diametral_disk_contains, split_protected_segment,
    ExactFaceCandidateClassifications,
};
use crate::refinement::control::checkpoint;
use crate::refinement::spatial;
use crate::refinement::{
    ExactFaceCandidateDisposition, ExactFaceRefinementCandidate, ExactFaceRefinementError,
    ExactFaceRefinementReason, ExactProtectedSegmentSplit,
};
use crate::{ExactFaceChartParameterization, ExactFacePslg, ExactFacePslgSegmentSource};
use runmat_geometry_core::GeometryAllocationScope;
use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::quality::spatial_index::SpatialEntry;
use runmat_meshing_core::MetricFieldRequest;
use std::collections::BTreeSet;

pub(crate) fn classify_exact_face_refinement_candidates_in_parameterization(
    candidates: &[ExactFaceRefinementCandidate],
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    parameterization: &ExactFaceChartParameterization,
    evaluator: &dyn ExactSurfaceEvaluator,
    control: &dyn GeometryEvaluationControl,
) -> Result<ExactFaceCandidateClassifications, ExactFaceRefinementError> {
    classify_with_context(
        candidates,
        pslg,
        parameterization,
        crate::ExactFaceGeometryContext::new(topology, request, evaluator, control),
    )
}
pub(crate) fn classify_with_context(
    candidates: &[ExactFaceRefinementCandidate],
    pslg: &ExactFacePslg,
    parameterization: &ExactFaceChartParameterization,
    context: crate::ExactFaceGeometryContext<'_>,
) -> Result<ExactFaceCandidateClassifications, ExactFaceRefinementError> {
    let control = context.control;
    let evaluator = context.evaluator;
    checkpoint(Some(control), &pslg.source_face_id)?;
    for candidate in candidates {
        checkpoint(Some(control), &pslg.source_face_id)?;
        if candidate.source_face_id != pslg.source_face_id
            || candidate.uv.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                &pslg.source_face_id,
                "candidate face or UV is inconsistent with the PSLG",
            ));
        }
    }
    let field = context
        .resolve_metric()
        .map_err(|error| metric(&pslg.source_face_id, error))?;
    let mut segment_metrics = Vec::with_capacity(pslg.segments.len());
    for (segment_index, segment) in pslg.segments.iter().enumerate() {
        checkpoint(Some(control), &pslg.source_face_id)?;
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
    let storage = GeometryAllocationScope::new(control);
    let extent = pslg
        .vertices
        .iter()
        .map(|vertex| vertex.uv)
        .chain(candidates.iter().map(|candidate| candidate.uv))
        .flatten()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let influence = spatial::build(
        segment_metrics.len(),
        segment_metrics
            .iter()
            .enumerate()
            .map(|(index, (_, first, second, metric))| SpatialEntry {
                bounds: super::influence::bounds(first.uv, second.uv, *metric, extent),
                payload: index,
            }),
        Some(&storage),
        &pslg.source_face_id,
    )?;
    let mut encroached_segments = BTreeSet::new();
    let mut insertions = Vec::new();
    let mut withdrawn = 0;
    for candidate in candidates {
        checkpoint(Some(control), &pslg.source_face_id)?;
        let mut encroached = false;
        let mut withdraw = false;
        let mut failure = None;
        influence.visit_bounds_with_checkpoint(
            spatial::point(candidate.uv),
            &mut |entry| {
                let (segment_index, first, second, metric) = &segment_metrics[entry.payload];
                let segment = &pslg.segments[*segment_index];
                if matches!(
                    candidate.reason,
                    ExactFaceRefinementReason::MetricAngle
                        | ExactFaceRefinementReason::PhysicalAspectRatio
                        | ExactFaceRefinementReason::BoundaryMetricSupport
                ) && matches!(segment.source, ExactFacePslgSegmentSource::ExactTrim { .. })
                {
                    // Shared exact curves already satisfy their independently validated metric-size
                    // contract. Face-owned shape refinement may split chart cuts, but it must not
                    // repeatedly rewrite a global curve merely to improve a local triangle angle or
                    // aspect ratio. Geometric-deviation and metric-size candidates remain hard and
                    // can request an exact-curve split. A shape point exactly on the curve (the
                    // circumcenter of a right triangle on it) can be neither inserted nor used to
                    // split the curve, so it is withdrawn; candidate selection keeps shape points
                    // a minimum metric spacing away from shared curves.
                    if runmat_meshing_core::predicate::orient2d([first.uv, second.uv, candidate.uv])
                        .is_ok_and(|sign| sign == runmat_meshing_core::PredicateSign::Zero)
                        && within_closed_bounds(candidate.uv, first.uv, second.uv)
                    {
                        withdraw = true;
                    }
                    return;
                }
                // A CDT permits interior points inside a protected edge's diametral
                // disk. Size-only face refinement must not repeatedly subdivide an
                // exact curve already within the unit metric-edge contract. Points
                // on the edge still require a shared split; geometric error and
                // oversized boundary edges retain the ordinary split path. The
                // final independent face-quality and protected-segment checks apply.
                if candidate.reason == ExactFaceRefinementReason::MetricEdgeLength
                    && matches!(segment.source, ExactFacePslgSegmentSource::ExactTrim { .. })
                    && metric
                        .squared_length([second.uv[0] - first.uv[0], second.uv[1] - first.uv[1]])
                        .is_ok_and(|length| length <= 1.0)
                    && runmat_meshing_core::predicate::orient2d([first.uv, second.uv, candidate.uv])
                        .is_ok_and(|sign| sign != runmat_meshing_core::PredicateSign::Zero)
                {
                    return;
                }
                let Some(encroached_by_candidate) =
                    metric_diametral_disk_contains(candidate.uv, first.uv, second.uv, *metric)
                else {
                    failure = Some(invalid(
                        &pslg.source_face_id,
                        "protected-segment encroachment measure is not finite",
                    ));
                    return;
                };
                if encroached_by_candidate {
                    encroached_segments.insert(*segment_index);
                    encroached = true;
                }
            },
            &mut || checkpoint(Some(control), &pslg.source_face_id),
        )?;
        if let Some(failure) = failure {
            return Err(failure);
        }
        if withdraw && !encroached {
            withdrawn += 1;
        } else if !encroached {
            insertions.push(candidate.clone());
        }
    }
    let mut protected_splits = Vec::new();
    let mut chart_cut_splits = Vec::new();
    for segment_index in encroached_segments {
        checkpoint(Some(control), &pslg.source_face_id)?;
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
            ExactFaceCandidateDisposition::Insert | ExactFaceCandidateDisposition::Withdraw => {
                return Err(invalid(
                    &pslg.source_face_id,
                    "encroached segment classification did not request a split",
                ));
            }
        }
    }
    checkpoint(Some(control), &pslg.source_face_id)?;
    Ok(ExactFaceCandidateClassifications {
        protected_splits,
        chart_cut_splits,
        insertions,
        withdrawn,
    })
}

fn within_closed_bounds(point: [f64; 2], first: [f64; 2], second: [f64; 2]) -> bool {
    (0..2).all(|axis| {
        point[axis] >= first[axis].min(second[axis]) && point[axis] <= first[axis].max(second[axis])
    })
}
