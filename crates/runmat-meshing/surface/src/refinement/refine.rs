use crate::{
    evaluate_exact_face_geometry_in_parameterization,
    validate_exact_face_geometry_in_parameterization, ExactFaceBoundary, ExactFaceChart,
    ExactFaceChartParameterization, ExactFaceGeometryContext, ExactFaceGeometryError,
};

use super::{
    classify_exact_face_refinement_candidate_in_parameterization,
    derive_exact_face_feature_collars, insert_exact_face_refinement_candidate,
    insert_validated_face_refinement_candidate, select_exact_face_refinement_candidate,
    split_exact_face_chart_cut, ExactFaceCandidateDisposition, ExactFaceRefinedMesh,
    ExactFaceRefinedTopology, ExactFaceRefinementContext, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind, ExactFaceRefinementOutcome, ExactFaceRefinementPolicy,
};

#[derive(Clone, Copy)]
pub(crate) enum ExactFaceRefinementDomain<'a> {
    ExactBoundary(&'a ExactFaceBoundary),
    Chart(&'a ExactFaceChart),
}

impl ExactFaceRefinementDomain<'_> {
    fn parameterization(&self) -> &ExactFaceChartParameterization {
        match self {
            Self::ExactBoundary(_) => &ExactFaceChartParameterization::EvaluatorParameters,
            Self::Chart(chart) => &chart.parameterization,
        }
    }
}

pub fn refine_exact_face_until_blocked(
    boundary: &ExactFaceBoundary,
    initial: &ExactFaceRefinedTopology,
    context: ExactFaceRefinementContext<'_>,
    policy: ExactFaceRefinementPolicy,
) -> Result<ExactFaceRefinementOutcome, ExactFaceRefinementError> {
    refine_exact_face_domain_until_blocked(
        ExactFaceRefinementDomain::ExactBoundary(boundary),
        initial,
        context,
        policy,
        0,
    )
    .map(|(outcome, _)| outcome)
}

pub(crate) fn refine_exact_face_domain_until_blocked(
    domain: ExactFaceRefinementDomain<'_>,
    initial: &ExactFaceRefinedTopology,
    context: ExactFaceRefinementContext<'_>,
    policy: ExactFaceRefinementPolicy,
    maximum_chart_cut_splits: u32,
) -> Result<(ExactFaceRefinementOutcome, u32), ExactFaceRefinementError> {
    policy.quality.validate().map_err(|error| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidQuality,
            &initial.pslg.source_face_id,
            error.to_string(),
        )
    })?;
    if policy.refinement.maximum_interior_insertions == 0 || policy.delaunay.validate().is_err() {
        return Err(ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidOptions,
            &initial.pslg.source_face_id,
            "refinement and Delaunay operational limits must be nonzero",
        ));
    }
    if matches!(domain, ExactFaceRefinementDomain::Chart(_)) && maximum_chart_cut_splits == 0 {
        return Err(ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidOptions,
            &initial.pslg.source_face_id,
            "chart-cut split hard limit must be nonzero",
        ));
    }
    let mut current = initial.clone();
    let mut insertion_count = 0u32;
    let mut chart_cut_split_count = 0u32;
    loop {
        let geometry = evaluate_exact_face_geometry_in_parameterization(
            &current.trimmed,
            &current.pslg,
            domain.parameterization(),
            ExactFaceGeometryContext::new(
                context.topology,
                context.metric_request,
                context.evaluator,
                context.geometry_control,
            ),
        )
        .map_err(map_geometry)?;
        validate_exact_face_geometry_in_parameterization(
            &geometry,
            &current.trimmed,
            &current.pslg,
            domain.parameterization(),
            ExactFaceGeometryContext::new(
                context.topology,
                context.metric_request,
                context.evaluator,
                context.geometry_control,
            ),
        )
        .map_err(map_geometry)?;
        let collars = derive_exact_face_feature_collars(&current.pslg, &geometry, policy.quality)?;
        let Some(candidate) = select_exact_face_refinement_candidate(
            &geometry,
            &current.pslg,
            &collars,
            policy.quality,
        )?
        else {
            return Ok((
                ExactFaceRefinementOutcome::Converged(Box::new(ExactFaceRefinedMesh {
                    topology: current,
                    geometry,
                    feature_collars: collars,
                    interior_insertion_count: insertion_count,
                })),
                chart_cut_split_count,
            ));
        };
        match classify_exact_face_refinement_candidate_in_parameterization(
            &candidate,
            &current.pslg,
            context.topology,
            context.metric_request,
            domain.parameterization(),
            context.evaluator,
            context.geometry_control,
        )? {
            ExactFaceCandidateDisposition::SplitProtectedSegment(split) => {
                return Ok((
                    ExactFaceRefinementOutcome::RequiresCurveSplit {
                        split,
                        completed_interior_insertions: insertion_count,
                    },
                    chart_cut_split_count,
                ));
            }
            ExactFaceCandidateDisposition::SplitChartCut(split) => match domain {
                ExactFaceRefinementDomain::ExactBoundary(_) => {
                    return Ok((
                        ExactFaceRefinementOutcome::RequiresChartCutSplit {
                            split,
                            completed_interior_insertions: insertion_count,
                        },
                        chart_cut_split_count,
                    ));
                }
                ExactFaceRefinementDomain::Chart(_) => {
                    if chart_cut_split_count >= maximum_chart_cut_splits {
                        return Err(ExactFaceRefinementError::new(
                            ExactFaceRefinementErrorKind::ResourceLimit,
                            &current.pslg.source_face_id,
                            "exact face refinement exceeded its chart-cut split hard limit",
                        ));
                    }
                    current = split_exact_face_chart_cut(
                        &current,
                        &split,
                        context.cancellation,
                        policy.delaunay,
                    )?;
                    chart_cut_split_count += 1;
                }
            },
            ExactFaceCandidateDisposition::Insert => {
                if insertion_count >= policy.refinement.maximum_interior_insertions {
                    return Err(ExactFaceRefinementError::new(
                        ExactFaceRefinementErrorKind::ResourceLimit,
                        &current.pslg.source_face_id,
                        "exact face refinement exceeded its interior insertion hard limit",
                    ));
                }
                current = match domain {
                    ExactFaceRefinementDomain::ExactBoundary(boundary) => {
                        insert_exact_face_refinement_candidate(
                            boundary,
                            &current,
                            &candidate,
                            context.cancellation,
                            policy.delaunay,
                        )?
                    }
                    ExactFaceRefinementDomain::Chart(_) => {
                        insert_validated_face_refinement_candidate(
                            &current,
                            &candidate,
                            context.cancellation,
                            policy.delaunay,
                        )?
                    }
                };
                insertion_count += 1;
            }
        }
    }
}

fn map_geometry(error: ExactFaceGeometryError) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Geometry(error.kind),
        &error.source_face_id,
        error.reason,
    )
}
