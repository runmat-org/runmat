use crate::{
    evaluate_exact_face_geometry_in_parameterization,
    validate_exact_face_geometry_in_parameterization, ExactFaceBoundary, ExactFaceChart,
    ExactFaceChartParameterization, ExactFaceGeometryContext, ExactFaceGeometryError,
};

use super::{
    classify_exact_face_refinement_candidates_in_parameterization,
    derive_exact_face_feature_collars, insert_exact_face_refinement_candidates,
    insert_validated_face_refinement_candidates, select_exact_face_refinement_candidates,
    split_exact_face_chart_cut, ExactFaceRefinedMesh, ExactFaceRefinedTopology,
    ExactFaceRefinementContext, ExactFaceRefinementError, ExactFaceRefinementErrorKind,
    ExactFaceRefinementOutcome, ExactFaceRefinementPolicy,
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
        let collars = derive_exact_face_feature_collars(&current.pslg, &geometry, policy.quality)?;
        let candidates = select_exact_face_refinement_candidates(
            &geometry,
            &current.pslg,
            &collars,
            policy.quality,
        )?;
        if candidates.is_empty() {
            // Independent validation deliberately belongs at the artifact
            // boundary. Re-evaluating every node and deviation sample after
            // every interior insertion turns refinement into a quadratic
            // sequence of full-prefix audits without protecting any additional
            // externally observable state.
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
            return Ok((
                ExactFaceRefinementOutcome::Converged(Box::new(ExactFaceRefinedMesh {
                    topology: current,
                    geometry,
                    feature_collars: collars,
                    interior_insertion_count: insertion_count,
                })),
                chart_cut_split_count,
            ));
        }

        let classifications = classify_exact_face_refinement_candidates_in_parameterization(
            &candidates,
            &current.pslg,
            context.topology,
            context.metric_request,
            domain.parameterization(),
            context.evaluator,
            context.geometry_control,
        )?;
        if !classifications.protected_splits.is_empty() {
            return Ok((
                ExactFaceRefinementOutcome::RequiresCurveSplits {
                    splits: classifications.protected_splits,
                    completed_interior_insertions: insertion_count,
                },
                chart_cut_split_count,
            ));
        }
        if let Some(split) = classifications.chart_cut_splits.into_iter().next() {
            match domain {
                ExactFaceRefinementDomain::ExactBoundary(_) => {
                    return Ok((
                        ExactFaceRefinementOutcome::RequiresChartCutSplit {
                            split: Box::new(split),
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
            }
        } else if !classifications.insertions.is_empty() {
            if insertion_count >= policy.refinement.maximum_interior_insertions {
                return Err(ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::ResourceLimit,
                    &current.pslg.source_face_id,
                    "exact face refinement exceeded its interior insertion hard limit",
                ));
            }
            let remaining = policy.refinement.maximum_interior_insertions - insertion_count;
            let mut candidates = classifications.insertions;
            candidates.truncate(remaining as usize);
            let (refined, inserted) = match domain {
                ExactFaceRefinementDomain::ExactBoundary(boundary) => {
                    insert_exact_face_refinement_candidates(
                        boundary,
                        &current,
                        &candidates,
                        context.cancellation,
                        policy.delaunay,
                    )?
                }
                ExactFaceRefinementDomain::Chart(_) => insert_validated_face_refinement_candidates(
                    &current,
                    &candidates,
                    context.cancellation,
                    policy.delaunay,
                )?,
            };
            current = refined;
            insertion_count += inserted;
        } else {
            return Err(ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::InvalidGeometry,
                &current.pslg.source_face_id,
                "refinement candidates produced no actionable disposition",
            ));
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
