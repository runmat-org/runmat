//! Accepted-state SQP controller, with separate proposed model, trial values,
//! derivative completion, original nonlinear KKT and observer boundaries.
use super::super::{
    evaluation::Evaluator,
    model::{Geometry, GeometryView},
};
use super::{Outcome, Parameters};
mod accepted;
mod proposal;
use crate::BuiltinResult;

pub(in crate::builtins::math::optim::fmincon) fn minimize<
    'geometry,
    'parameters,
    'controls,
    'evaluator,
    'source,
>(
    x: Vec<f64>,
    geometry: &'geometry Geometry,
    parameters: &'parameters Parameters<'controls>,
    evaluator: &'evaluator mut Evaluator<'source>,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>>
       + use<'geometry, 'parameters, 'controls, 'evaluator, 'source> {
    minimize_with(x, geometry, parameters, evaluator, super::Strategy::Conic)
}

pub(in crate::builtins::math::optim::fmincon) async fn minimize_with(
    x: Vec<f64>,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
    strategy: super::Strategy,
) -> BuiltinResult<Outcome> {
    let out = initialize(x, geometry, parameters, evaluator, strategy).await?;
    accepted::run(out, geometry, parameters, evaluator, strategy).await
}

pub(in crate::builtins::math::optim::fmincon) async fn minimize_view(
    x: Vec<f64>,
    geometry: &GeometryView<'_>,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
    strategy: super::Strategy,
) -> BuiltinResult<Outcome> {
    let out = initialize_view(x, geometry, parameters, evaluator, strategy).await?;
    accepted::run_view(out, geometry, parameters, evaluator, strategy).await
}

/// The existing selected feasibility owner stages these two concrete phases.
/// Ordinary numerical callers retain minimize/minimize_with and need no service.
pub(in crate::builtins::math::optim::fmincon) async fn minimize_staged(
    runtime: &crate::context::RuntimeContext,
    x: Vec<f64>,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
    strategy: super::Strategy,
) -> BuiltinResult<Outcome> {
    let original = &mut *evaluator;
    let out = crate::execution::admit_native_hostcall_with_contextual_error(runtime, move || {
        initialize(x, geometry, parameters, original, strategy)
    })?
    .await?;
    crate::execution::admit_native_hostcall_with_contextual_error(runtime, move || {
        accepted::run(out, geometry, parameters, evaluator, strategy)
    })?
    .await
}

async fn initialize(
    x: Vec<f64>,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
    strategy: super::Strategy,
) -> BuiltinResult<Outcome> {
    initialize_view(x, &geometry.view(), parameters, evaluator, strategy).await
}

async fn initialize_view(
    x: Vec<f64>,
    geometry: &GeometryView<'_>,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
    strategy: super::Strategy,
) -> BuiltinResult<Outcome> {
    if evaluator.semi_infinite.is_some() && !matches!(strategy, super::Strategy::WorkingSet) {
        return Err(super::super::invalid(
            "semi-infinite SQP requires the actual working-set quadratic strategy",
        ));
    }
    let mut out = super::initialization::initialize(x, geometry, parameters, evaluator).await?;
    if matches!(strategy, super::Strategy::WorkingSet) {
        out.algorithm = runmat_builtins::OptimAlgorithm::ActiveSet;
    }
    Ok(out)
}
