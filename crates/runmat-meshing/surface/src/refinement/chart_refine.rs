use crate::{ExactFaceChart, ExactFaceChartConstrainedDomain};

use super::{
    refine::{refine_exact_face_domain_until_blocked, ExactFaceRefinementDomain},
    ExactFaceChartRefinedMesh, ExactFaceChartRefinementOptions, ExactFaceChartRefinementOutcome,
    ExactFaceRefinedTopology, ExactFaceRefinementContext, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind, ExactFaceRefinementOutcome, ExactFaceRefinementPolicy,
};

pub fn refine_exact_face_chart_until_blocked(
    chart: &ExactFaceChart,
    initial: &ExactFaceChartConstrainedDomain,
    context: ExactFaceRefinementContext<'_>,
    policy: ExactFaceRefinementPolicy,
    chart_options: ExactFaceChartRefinementOptions,
) -> Result<ExactFaceChartRefinementOutcome, ExactFaceRefinementError> {
    if chart.chart_id == runmat_meshing_core::StableDigest::ZERO
        || chart_options.maximum_chart_cut_splits == 0
    {
        return Err(ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidOptions,
            &chart.source_face_id,
            "chart identity and chart-cut split hard limit must be nonzero",
        ));
    }
    if initial.chart_id != chart.chart_id || chart.source_face_id != chart.pslg.source_face_id {
        return Err(ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidGeometry,
            &chart.source_face_id,
            "chart and constrained-domain identities are inconsistent",
        ));
    }
    crate::exact_cdt::validate_face_delaunay_topology(
        &initial.delaunay,
        &chart.pslg,
        context.cancellation,
        policy.delaunay,
    )
    .map_err(|error| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::Delaunay(error.kind),
            &error.source_face_id,
            error.reason,
        )
    })?;
    let topology = ExactFaceRefinedTopology {
        pslg: chart.pslg.clone(),
        constrained: initial.constrained.clone(),
        trimmed: initial.trimmed.clone(),
    };
    let (outcome, chart_cut_split_count) = refine_exact_face_domain_until_blocked(
        ExactFaceRefinementDomain::Chart(chart),
        &topology,
        context,
        policy,
        chart_options.maximum_chart_cut_splits,
    )?;
    match outcome {
        ExactFaceRefinementOutcome::Converged(mesh) => Ok(
            ExactFaceChartRefinementOutcome::Converged(Box::new(ExactFaceChartRefinedMesh {
                chart_id: chart.chart_id,
                mesh: *mesh,
                chart_cut_split_count,
            })),
        ),
        ExactFaceRefinementOutcome::RequiresCurveSplit {
            split,
            completed_interior_insertions,
        } => Ok(ExactFaceChartRefinementOutcome::RequiresCurveSplit {
            chart_id: chart.chart_id,
            split,
            completed_interior_insertions,
            completed_chart_cut_splits: chart_cut_split_count,
        }),
        ExactFaceRefinementOutcome::RequiresChartCutSplit { .. } => {
            Err(ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::InvalidGeometry,
                &chart.source_face_id,
                "chart refinement returned an unapplied face-owned cut split",
            ))
        }
    }
}
