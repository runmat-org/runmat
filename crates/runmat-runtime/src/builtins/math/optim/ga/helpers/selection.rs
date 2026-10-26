use super::{
    super::{
        invalid,
        operators::{ranking, selection},
        random, POLICY,
    },
    admission,
};
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_builtins::OptimGeneticSelection as Kind;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

async fn execute(
    kind: Kind,
    runtime: Option<&RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() != 3 && !(kind == Kind::Tournament && args.len() == 4) {
        return Err(invalid(
            "selection requires expectation,nParents,options and optional tournament size",
        ));
    }
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_SELECTION_HELPER_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        kind.name(),
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = admission::begin(kind.name(), runtime)?;
    cooperation.completed(0).await?;
    admission::vector_metadata(&args[0], runtime)?;
    admission::count_metadata(&args[1], runtime)?;
    admission::options(&args[2])?;
    if let Some(size) = args.get(3) {
        admission::count_metadata(size, runtime)?;
    }
    let count = admission::count(&args[1], "nParents", &mut cooperation).await?;
    let size = if let Some(size) = args.get(3) {
        admission::count(size, "tournament size", &mut cooperation).await?
    } else {
        4
    };
    if size < 2 {
        return Err(invalid("tournament size must be at least two"));
    }
    let weights = admission::vector(args[0].clone(), &mut cooperation).await?;
    let mut positive = false;
    for &weight in &weights {
        if !weight.is_finite() || weight < 0. {
            return Err(invalid(
                "selection expectations must be finite and nonnegative",
            ));
        }
        positive |= weight > 0.;
        cooperation.completed(1).await?;
    }
    let selected = if count == 0 {
        POLICY.capacity(0, runtime)?
    } else {
        if !positive {
            return Err(invalid(
                "positive nParents requires positive selection weight",
            ));
        }
        match kind {
            Kind::StochasticUniform => {
                let draws = random::uniform(1, &mut cooperation).await?;
                ranking::stochastic_uniform(&weights, count, draws[0], &mut cooperation).await?
            }
            Kind::Remainder => {
                let remainder =
                    selection::Remainder::prepare(&weights, count, &mut cooperation).await?;
                let draws = random::uniform(remainder.remaining(), &mut cooperation).await?;
                remainder.finish(&draws, &mut cooperation).await?
            }
            Kind::Uniform => {
                let draws = random::uniform(count, &mut cooperation).await?;
                selection::uniform(weights.len(), &draws, &mut cooperation).await?
            }
            Kind::Roulette => {
                let draws = random::uniform(count, &mut cooperation).await?;
                selection::roulette(&weights, &draws, &mut cooperation).await?
            }
            Kind::Tournament => {
                let length = count
                    .checked_mul(size)
                    .ok_or_else(|| invalid("tournament random extent overflow"))?;
                let draws = random::uniform(length, &mut cooperation).await?;
                ranking::tournament(&weights, &draws, size, &mut cooperation).await?
            }
        }
    };
    let mut values = POLICY.capacity(selected.len(), runtime)?;
    for index in selected {
        let index = index
            .checked_add(1)
            .ok_or_else(|| invalid("selected parent index overflow"))?;
        let index = u64::try_from(index)
            .ok()
            .and_then(|index| runmat_value::IntValue::U64(index).to_exact_f64())
            .ok_or_else(|| invalid("selected parent cannot be represented exactly as double"))?;
        values.push(index);
        cooperation.completed(1).await?;
    }
    admission::finish(values, true, runtime)
}

#[runtime_builtin(
    name = "selectionstochunif",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::selection"
)]
async fn stochastic_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::StochasticUniform, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "selectionremainder",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::selection"
)]
async fn remainder_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Remainder, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "selectionuniform",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::selection"
)]
async fn uniform_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Uniform, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "selectionroulette",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::selection"
)]
async fn roulette_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Roulette, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

#[runtime_builtin(
    name = "selectiontournament",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::selection"
)]
async fn tournament_registered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Tournament, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
