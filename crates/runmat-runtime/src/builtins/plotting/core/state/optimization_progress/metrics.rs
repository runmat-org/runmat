//! Typed measured metric supplied by the actual observer generation. The
//! absence of an objective is not zero objective value, and feasibility uses
//! the selected nonlinear tolerance rather than a solver-name guess.
use super::*;
pub(crate) struct ConstrainedPlotSample {
    pub objective: Option<f64>,
    pub maximum: Option<f64>,
    pub tolerance: f64,
    pub feasible_count: Option<usize>,
}
pub(crate) fn update_constrained_metric(
    phase: OutputState,
    iteration: f64,
    sample: ConstrainedPlotSample,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if !sample.tolerance.is_finite()
        || sample.tolerance < 0.0
        || sample.maximum.is_some_and(|v| !v.is_finite())
        || runmat_types::decode_shape_dimension(iteration).is_err()
    {
        return Err(failure("invalid constrained chart metric or iteration"));
    }
    admission(runtime)?;
    let _scope = runtime.map(RuntimeContext::enter);
    let resolved = resolve_axes_target(PropertyMutationTarget::alias().with_runtime(runtime), None)
        .map_err(graphics_failure)?;
    let title = if sample.objective.is_none() && sample.maximum.is_some() {
        Some(sample.feasible_count.map_or_else(
            || "Nonlinear Constraint Value".to_owned(),
            |n| format!("Feasible solutions: {n}"),
        ))
    } else {
        None
    };
    let annotation = title.as_ref().map(|title| ChartAnnotation {
        title,
        y_label: "Maximum nonlinear constraint value",
    });
    let Some(value) = sample.objective.or(sample.maximum) else {
        for series in [
            OptimizationSeries::NonlinearFeasible,
            OptimizationSeries::NonlinearInfeasible,
            OptimizationSeries::NonlinearBest,
        ] {
            update_series_payload(
                series,
                phase,
                OptimizationPlotData::Empty,
                runtime,
                Some(&resolved),
                annotation,
            )?;
        }
        return Ok(());
    };
    let feasible = sample.maximum.is_none_or(|v| v <= sample.tolerance);
    for (series, value) in [
        (
            OptimizationSeries::NonlinearFeasible,
            if feasible { value } else { f64::NAN },
        ),
        (
            OptimizationSeries::NonlinearInfeasible,
            if feasible { f64::NAN } else { value },
        ),
        (
            OptimizationSeries::NonlinearBest,
            if feasible { value } else { f64::NAN },
        ),
    ] {
        update_series_payload(
            series,
            phase,
            OptimizationPlotData::Sample {
                point: Vec::new(),
                iteration,
                value,
            },
            runtime,
            Some(&resolved),
            annotation,
        )?;
    }
    Ok(())
}
