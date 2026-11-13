//! Predefined crossover calls share the controller's numerical proposal owner.
use super::{
    super::{common, invalid, numeric, operators::crossover, random, POLICY},
    admission,
};
use crate::{context::RuntimeContext, BuiltinResult};
use runmat_builtins::OptimGeneticCrossover as Kind;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
#[cfg(test)]
#[path = "crossover/tests.rs"]
mod tests;

async fn execute(
    kind: Kind,
    runtime: Option<&RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let accepts_ratio = matches!(kind, Kind::Intermediate | Kind::Heuristic);
    if args.len() != 6 && !(accepts_ratio && args.len() == 7) {
        return Err(invalid("crossover requires parents,options,nvars,problem,scores,population and an optional supported ratio"));
    }
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_CROSSOVER_HELPER_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        kind.name(),
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = admission::begin(kind.name(), runtime)?;
    cooperation.completed(0).await?;
    admission::vector_metadata(&args[0], runtime)?;
    admission::options(&args[1])?;
    admission::count_metadata(&args[2], runtime)?;
    if !matches!(&args[3], Value::Struct(_)) {
        return Err(invalid("crossover problem must be a scalar structure"));
    }
    if kind == Kind::Heuristic {
        admission::vector_metadata(&args[4], runtime)?;
    }
    numeric::preflight(
        &args[5],
        "crossover population",
        common::constrained_numeric::Geometry::MatrixAny,
        runtime,
    )?;
    if let Some(ratio) = args.get(6) {
        admission::vector_metadata(ratio, runtime)?;
    }
    let variables = admission::count(&args[2], "nvars", &mut cooperation).await?;
    if variables == 0 {
        return Err(invalid("nvars must be positive"));
    }
    let parents = admission::vector(args[0].clone(), &mut cooperation).await?;
    if parents.len() % 2 != 0 {
        return Err(invalid(
            "crossover requires one pair of parent indices per child",
        ));
    }
    let common::constrained_numeric::Numeric::Real { shape, values } =
        common::constrained_numeric::read(
            args[5].clone(),
            "crossover population",
            common::constrained_numeric::Geometry::MatrixAny,
            &numeric::INPUT_POLICY,
            &mut cooperation,
        )
        .await?
    else {
        return Err(invalid("crossover requires a real population"));
    };
    if shape.len() != 2 || shape[1] != variables {
        return Err(invalid("population must have nvars columns"));
    }
    let members = shape[0];
    for value in &values {
        if !value.is_finite() {
            return Err(invalid("crossover population must be finite"));
        }
        cooperation.completed(1).await?;
    }
    let mut indices = POLICY.capacity(parents.len(), runtime)?;
    for parent in parents {
        if !parent.is_finite() || parent < 1. || parent.fract() != 0. || parent >= usize::MAX as f64
        {
            return Err(invalid(
                "parents must contain exact one-based population row indices",
            ));
        }
        let index = parent as usize - 1;
        if index >= members {
            return Err(invalid("parent index exceeds population row count"));
        }
        indices.push(index);
        cooperation.completed(1).await?;
    }
    let scores = if kind == Kind::Heuristic {
        let scores = admission::vector(args[4].clone(), &mut cooperation).await?;
        if scores.len() != members || scores.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "heuristic crossover requires one finite score per population row",
            ));
        }
        Some(scores)
    } else {
        None
    };
    let ratio = if let Some(value) = args.get(6) {
        let values = admission::vector(value.clone(), &mut cooperation).await?;
        if values.is_empty() {
            return Err(invalid(
                "an explicitly supplied crossover ratio must be nonempty",
            ));
        }
        values
    } else if kind == Kind::Intermediate {
        vec![1.]
    } else {
        Vec::new()
    };
    if ratio.iter().any(|v| !v.is_finite() || *v < 0.)
        || (!ratio.is_empty()
            && ratio.len() != 1
            && (kind != Kind::Intermediate || ratio.len() != variables))
    {
        return Err(invalid("ratio must be nonnegative finite and scalar, or one per variable for intermediate crossover"));
    }
    let children = indices.len() / 2;
    let length = children
        .checked_mul(variables)
        .ok_or_else(|| invalid("crossover output extent overflow"))?;
    let mut output = POLICY.capacity(length, runtime)?;
    output.resize(length, 0.);
    let mut first = POLICY.capacity(variables, runtime)?;
    let mut second = POLICY.capacity(variables, runtime)?;
    for (child, pair) in indices.chunks_exact(2).enumerate() {
        first.clear();
        second.clear();
        for column in 0..variables {
            first.push(values[pair[0] + members * column]);
            second.push(values[pair[1] + members * column]);
            cooperation.completed(1).await?;
        }
        let parameters = crossover::Parameters {
            kind,
            ratio: &ratio,
            first_is_better: scores
                .as_ref()
                .is_none_or(|scores| scores[pair[0]] <= scores[pair[1]]),
        };
        let draws = random::uniform(
            crossover::draw_count(&parameters, variables)?,
            &mut cooperation,
        )
        .await?;
        let point = crossover::child(parameters, &first, &second, &draws, &mut cooperation)
            .await?
            .ok_or_else(|| invalid("crossover child is outside finite binary64 arithmetic"))?;
        for (column, coordinate) in point.into_iter().enumerate() {
            output[child + children * column] = coordinate;
            cooperation.completed(1).await?;
        }
    }
    if crate::output_count::current_output_count_in(runtime) == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    Tensor::new(output, vec![children, variables])
        .map(Value::Tensor)
        .map_err(invalid)
}

#[runtime_builtin(
    name = "crossoverscattered",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::crossover"
)]
async fn scattered(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Scattered, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
#[runtime_builtin(
    name = "crossoversinglepoint",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::crossover"
)]
async fn single(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::SinglePoint, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
#[runtime_builtin(
    name = "crossovertwopoint",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::crossover"
)]
async fn two(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::TwoPoint, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
#[runtime_builtin(
    name = "crossoverintermediate",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::crossover"
)]
async fn intermediate(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Intermediate, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
#[runtime_builtin(
    name = "crossoverarithmetic",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::crossover"
)]
async fn arithmetic(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Arithmetic, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
#[runtime_builtin(
    name = "crossoverheuristic",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::crossover"
)]
async fn heuristic(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Heuristic, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
#[runtime_builtin(
    name = "crossoverlaplace",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga::helpers::crossover"
)]
async fn laplace(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    execute(Kind::Laplace, runtime.as_ref(), args)
        .await
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}
