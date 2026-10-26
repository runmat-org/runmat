use super::{
    super::{
        invalid,
        operators::{
            ranking::{self, Score},
            scaling, shift_linear,
        },
        options, POLICY,
    },
    admission,
};
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_builtins::OptimGeneticScaling as Kind;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

async fn execute(
    kind: Kind,
    runtime: Option<&RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() != 2 && !(matches!(kind, Kind::Top | Kind::ShiftLinear) && args.len() == 3) {
        return Err(invalid(
            "fitness scaling requires scores,nParents and its optional scaling parameter",
        ));
    }
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_SCALING_HELPER_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        kind.name(),
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = admission::begin(kind.name(), runtime)?;
    cooperation.completed(0).await?;
    admission::vector_metadata(&args[0], runtime)?;
    admission::count_metadata(&args[1], runtime)?;
    if let Some(parameter) = args.get(2) {
        options::admission::numeric_metadata(
            options::Owner::Solver,
            options::admission::Role::Real,
            parameter,
            runtime,
        )?;
    }
    let parents = admission::count(&args[1], "nParents", &mut cooperation).await?;
    let parameter = if let Some(parameter) = args.get(2) {
        let parameter =
            crate::dispatcher::gather_if_needed_in_context_async(parameter, runtime).await?;
        Some(options::real(options::Owner::Solver, &parameter)?)
    } else {
        None
    };
    let raw = admission::vector(args[0].clone(), &mut cooperation).await?;
    let mut scores = POLICY.capacity(raw.len(), runtime)?;
    for &score in &raw {
        scores.push(if score.is_finite() {
            Score::Defined(score)
        } else {
            Score::Undefined
        });
        cooperation.completed(1).await?;
    }
    if raw.is_empty() && parents != 0 {
        return Err(invalid("positive nParents requires population scores"));
    }
    let order = ranking::order(&scores, &mut cooperation).await?;
    let expectation = match kind {
        Kind::Rank => ranking::rank_expectation(&order, parents, &mut cooperation).await?,
        Kind::Proportional => scaling::proportional(&raw, parents, &mut cooperation).await?,
        Kind::Top => {
            scaling::top(&order, parameter.unwrap_or(0.4), parents, &mut cooperation).await?
        }
        Kind::ShiftLinear => {
            shift_linear::expectation(
                &raw,
                &order,
                parameter.unwrap_or(2.),
                parents,
                &mut cooperation,
            )
            .await?
        }
    };
    scaling::validate(&expectation, parents, &mut cooperation).await?;
    admission::finish(expectation, false, runtime)
}

#[runtime_builtin(
    name = "fitscalingrank",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::scaling"
)]
async fn rank_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Rank, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "fitscalingprop",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::scaling"
)]
async fn proportional_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Proportional, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "fitscalingtop",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::scaling"
)]
async fn top_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Top, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "fitscalingshiftlinear",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::scaling"
)]
async fn shift_linear_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::ShiftLinear, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
