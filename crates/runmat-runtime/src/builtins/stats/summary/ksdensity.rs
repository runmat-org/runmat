//! Normal-kernel univariate density estimates with explicit public frontiers.
use super::inferential::{self, invalid, Cooperation};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
mod bandwidth;
mod engine;
mod options;
mod presentation;
#[cfg(test)]
mod tests;
const NAME: &str = "ksdensity";

#[runtime_builtin(
    name = "ksdensity",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::ksdensity"
)]
async fn ksdensity_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(runtime.as_ref(), args).await
}
async fn evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let count = crate::output_count::current_output_count().unwrap_or(1);
    if args.is_empty() || count > 3 {
        return Err(invalid(
            NAME,
            "expected sample data and at most three outputs",
        ));
    }
    crate::context::host_allocation::require_unbounded_workspace(runtime,NAME,
        "finite host-memory execution is unsupported; KDE sample, sorting, evaluation and result workspaces have no complete allocation bound")?;
    inferential::admit_automatic_host_arguments(NAME, args.view())?;
    let _sources = args.fork();
    let (axes, args) = presentation::split(args)?;
    presentation::admit(count, &axes)?;
    if args.is_empty() {
        return Err(invalid(
            NAME,
            "sample data is required after the axes target",
        ));
    }
    inferential::real_sample(NAME, &args[0])?;
    let second = args.get(1).filter(|value| {
        crate::builtins::common::validation::value_text_scalar_string(value).is_none()
    });
    if let Some(points) = second {
        inferential::real_sample(NAME, points)?;
    }
    let points = second.filter(|value| !empty_points(value));
    let options = options::Options::parse(
        args.view(),
        if second.is_some() { 2 } else { 1 },
        points.is_some(),
        runtime,
    )
    .await?;
    let mut cooperation = Cooperation::new(runtime);
    cooperation.completed(0).await?;
    let sample = inferential::gather(NAME, &args[0], runtime).await?;
    cooperation.completed(0).await?;
    if !vector(&sample) {
        return Err(invalid(
            NAME,
            "only univariate vectors are supported; bivariate KDE is unavailable",
        ));
    }
    let mut data = buffer(sample.len(), runtime)?;
    for offset in (0..sample.len()).step_by(1024) {
        let end = offset.saturating_add(1024).min(sample.len());
        for index in offset..end {
            let value = crate::builtins::common::tensor::tensor_value_f64(&sample, index);
            if value.is_infinite() {
                return Err(invalid(NAME, "sample observations must be finite or NaN"));
            }
            if !value.is_nan() {
                data.push(value);
            }
        }
        cooperation.completed(end - offset).await?;
    }
    if data.is_empty() {
        return Err(invalid(
            NAME,
            "at least one non-NaN observation is required",
        ));
    }
    let width = match options.bandwidth {
        Some(width) => width,
        None => bandwidth::normal_reference(&mut data, runtime, &mut cooperation).await?,
    };
    let points = match points {
        Some(points) => {
            let points = inferential::gather(NAME, points, runtime).await?;
            if !vector(&points) {
                return Err(invalid(NAME, "evaluation points must be a vector"));
            }
            points
        }
        None => engine::grid(&data, width, options.points, runtime, &mut cooperation).await?,
    };
    cooperation.completed(0).await?;
    let estimates = engine::evaluate(
        &data,
        &points,
        width,
        options.function,
        runtime,
        &mut cooperation,
    )
    .await?;
    let estimates =
        Tensor::new(estimates, points.shape().to_vec()).map_err(|error| invalid(NAME, error))?;
    if count == 0 {
        presentation::show(axes, points, estimates, runtime).await?;
        return Ok(Value::output_list(Vec::new()));
    }
    let mut outputs = Vec::with_capacity(count);
    outputs.push(Value::Tensor(estimates));
    if count >= 2 {
        outputs.push(Value::Tensor(points));
    }
    if count >= 3 {
        outputs.push(Value::Num(width));
    }
    cooperation.completed(0).await?;
    Ok(Value::output_list(outputs))
}
fn vector(value: &Tensor) -> bool {
    value.shape().iter().filter(|&&extent| extent != 1).count() <= 1
}
fn empty_points(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => value.is_empty(),
        Value::GpuTensor(handle) => handle.shape().contains(&0),
        _ => false,
    }
}

fn buffer(
    count: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<f64>> {
    runmat_value::exact_capacity_vec(count).map_err(|source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            runtime,
            "RunMat:ksdensity:AllocationFailed",
            "KDE numeric workspace geometry or allocation failed",
            crate::context::host_allocation::HostAllocationError::Numeric(source),
        )
    })
}
