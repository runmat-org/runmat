//! Invocation-owned feasible pattern search; callback and mesh owners differ.
use crate::builtins::math::optim::common::{constrained_numeric as numeric, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
mod augmented;
mod cache;
mod cone;
mod controls;
mod data;
mod evaluation;
pub(crate) mod helpers;
mod input;
mod iteration;
mod linear;
mod mads;
mod mesh;
mod normal;
mod nups;
mod observation;
pub(super) mod option_metadata;
mod options;
mod phase_one;
mod poll;
mod publication;
mod random;
mod run;
mod scaling;
mod search;
mod state;
mod timing;
#[cfg(test)]
mod tests;
const NAME: &str = "patternsearch";
const IDENTITY: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: NAME };

const NUMERIC: numeric::Policy = numeric::Policy {
    owner: IDENTITY,
    invalid: None,
    numeric: Some(&runmat_builtins::PATTERNSEARCH_NUMERIC_EXTENSION),
    single: Some(&runmat_builtins::PATTERNSEARCH_SINGLE_EXTENSION),
    resident: Some(&runmat_builtins::PATTERNSEARCH_RESIDENT_EXTENSION),
    sparse: None,
};
const OPTIONS: option_metadata::Policy = option_metadata::Policy {
    name: NAME,
    error: &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
    integer: &runmat_builtins::PATTERNSEARCH_INTEGER_OPTION_EXTENSION,
    resident: Some(&runmat_builtins::PATTERNSEARCH_RESIDENT_EXTENSION),
};

#[runtime_builtin(
    name = "patternsearch",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::patternsearch"
)]
async fn registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    execute(runtime.as_ref(), args).await.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME,
        "finite host-memory execution is unsupported; direct-search geometry, callbacks, cone and augmented-Lagrangian workspaces have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::PATTERNSEARCH_DESCRIPTOR,
        &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    let input = input::Input::parse(args.view())?;
    data::preflight(&input, &NUMERIC, runtime)?;
    let shape = numeric::shape(input.initial).ok_or_else(|| invalid("x0 must be a real vector"))?;
    let n = shape
        .iter()
        .try_fold(1usize, |n, &axis| n.checked_mul(axis))
        .ok_or_else(|| invalid("variable extent overflow"))?;
    if n == 0 {
        return Err(invalid("x0 must be nonempty"));
    }
    let active = |value: &runmat_value::Value| {
        numeric::shape(value).is_some_and(|shape| !shape.contains(&0))
    };
    let equalities = input.equality.is_some_and(|(_, rhs)| active(rhs));
    let linear = equalities
        || input.inequality.is_some_and(|(_, rhs)| active(rhs))
        || input.lower.is_some_and(active)
        || input.upper.is_some_and(active);
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, runtime);
    cooperation.completed(0).await?;
    let options = options::Options::prepare(
        input.options,
        n,
        equalities,
        linear,
        input.constraints.is_some(),
        OPTIONS,
        &mut cooperation,
    )
    .await?;
    let retain_search_window = options
        .controls
        .search
        .is_some_and(|search| matches!(search, search::Search::Latin { .. }))
        || crate::builtins::math::optim::common::output_callback::callback_count(
            options.callback(runmat_builtins::OptimOptionField::OutputFcn),
            crate::builtins::math::optim::common::output_callback::CallbackForm::Handles,
        )? > 0;
    let retain_original_bounds = options.controls.search.is_some_and(|search| {
        matches!(search, search::Search::Rbf | search::Search::Genetic { .. })
    })
        || crate::builtins::math::optim::common::output_callback::callback_count(
            options.callback(runmat_builtins::OptimOptionField::OutputFcn),
            crate::builtins::math::optim::common::output_callback::CallbackForm::Handles,
        )? > 0;
    let prepared = data::prepare(
        &input,
        &NUMERIC,
        retain_search_window,
        retain_original_bounds,
        &mut cooperation,
    )
    .await?;
    let problem_type = if input.constraints.is_some() {
        "nonlinearconstr"
    } else if prepared.linearly_constrained {
        "linearconstraints"
    } else if prepared.bound_constrained {
        "boundconstraints"
    } else {
        "unconstrained"
    };
    let scalar = prepared.scalar;
    let constrained = prepared.linear.matrix.m > 0 || input.constraints.is_some();
    let domain = observation::Domain {
        equalities: prepared.linear.equalities > 0,
        linear: prepared.linear.matrix.m > 0,
        nonlinear: input.constraints.is_some(),
        policy: OPTIONS,
    };
    // Admit the actual solver invocation before constructing its large
    // controller future. This is one boundary per solve, not per trial point.
    let selected_runtime = cooperation.runtime().cloned();
    let (outcome, rng, options) =
        crate::execution::rooted_continuation::admit_optional_hostcall_with_contextual_error(
            selected_runtime.as_ref(),
            || run::solve(&input, prepared, options, domain, NUMERIC, &mut cooperation),
        )?
        .await?;
    publication::finalize(
        outcome,
        rng,
        input.objective,
        &shape,
        scalar,
        constrained,
        problem_type,
        &options.controls,
        &mut cooperation,
    )
    .await
}
fn invalid(message: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("patternsearch: {message}"))
        .with_builtin(NAME)
        .with_identifier("RunMat:patternsearch:InvalidArgument")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn buffer<T>(capacity: usize) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(capacity).map_err(|source| {
        build_runtime_error("patternsearch: workspace allocation failed")
            .with_builtin(NAME)
            .with_identifier("RunMat:patternsearch:AllocationFailed")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    })
}
fn push<T>(values: &mut Vec<T>, value: T) -> BuiltinResult<()> {
    if values.len() == values.capacity() {
        values.try_reserve(1).map_err(|source| {
            build_runtime_error("patternsearch: workspace growth failed")
                .with_builtin(NAME)
                .with_identifier("RunMat:patternsearch:AllocationFailed")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(source)
                .build()
        })?;
    }
    values.push(value);
    Ok(())
}
