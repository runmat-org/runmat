//! Bound-reflective nonlinear Gauss--Newton trust region; LM remains separate.
use super::*;
use crate::builtins::math::optim::common::trust_quadratic;
use runmat_builtins::OptimDisplay;
mod controls;
mod cg;
mod normal;
mod geometry;
mod direct;
mod trajectory;
mod iteration;
use geometry::Geometry;
struct Configured<'a, E> {
    name: &'a str,
    evaluator: &'a mut E,
    point: initial::Seed,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
    display: OptimDisplay,
    controls: controls::Controls,
}
pub(super) fn solve<'a, E: LeastSquaresEvaluator + 'a>(
    name: &'a str,
    evaluator: &'a mut E,
    point: initial::Seed,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
    display: OptimDisplay,
    source: Option<&'a runmat_value::StructValue>,
) -> impl Future<Output = BuiltinResult<LeastSquaresResult>> + 'a {
    crate::execution::after_result(
        configure(name, evaluator, point, bounds, options, display, source),
        |configured| {
            let Configured {
                name,
                evaluator,
                point,
                bounds,
                options,
                display,
                controls,
            } = configured;
            iteration::solve(name, evaluator, point, bounds, options, display, controls)
        },
    )
}
async fn configure<'a, E: LeastSquaresEvaluator>(
    name: &'a str,
    evaluator: &'a mut E,
    point: initial::Seed,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
    display: OptimDisplay,
    source: Option<&'a runmat_value::StructValue>,
) -> BuiltinResult<Configured<'a, E>> {
    let controls = controls::prepare(options, source, point.x.len(), evaluator).await?;
    Ok(Configured {
        name,
        evaluator,
        point,
        bounds,
        options,
        display,
        controls,
    })
}
fn invalid(detail: String) -> crate::RuntimeError {
    optim_error("lsqnonlin", detail)
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("lsqnonlin: reflective workspace allocation failed")
        .with_builtin("lsqnonlin")
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn policy() -> trust_quadratic::Policy {
    trust_quadratic::Policy {
        identity: runmat_builtins::LSQNONLIN_CATALOG_ENTRY.identity,
        value_source: None,
        ownership_identifier: "RunMat:lsqnonlin:ReflectiveOwnership",
        invalid,
        allocation,
    }
}
async fn zeros(
    count: usize,
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<Vec<f64>> {
    zeros_with(count, cooperation(evaluator)?).await
}
async fn zeros_with(count: usize, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut values = runmat_value::exact_capacity_vec(count).map_err(allocation)?;
    while values.len() < count {
        let end = values.len().saturating_add(1024).min(count);
        let work = end - values.len();
        values.resize(end, 0.);
        c.completed(work as u64).await?;
    }
    Ok(values)
}
fn cooperation(
    evaluator: &mut impl LeastSquaresEvaluator,
) -> BuiltinResult<&mut ScalarCooperation> {
    evaluator.cooperation().ok_or_else(|| {
        invalid("reflective iteration requires its original cooperative evaluator".into())
    })
}
use super::display::Event;
fn emit(
    evaluator: &mut impl LeastSquaresEvaluator,
    mode: OptimDisplay,
    event: Event,
    line: impl FnOnce() -> String,
) -> BuiltinResult<()> {
    let runtime = evaluator.cooperation().and_then(|c| c.runtime());
    super::display::emit(runtime, mode, event, line)
}
