//! One-sample and paired Student-t tests, with explicit host resource frontier.
use super::inferential::{self, invalid, Cooperation};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Tensor, Value};

mod engine;
mod options;
#[cfg(test)]
mod tests;
const NAME: &str = "ttest";

#[runtime_builtin(
    name = "ttest",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::ttest"
)]
async fn ttest_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(runtime.as_ref(), args).await
}

pub(super) async fn evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let count = crate::output_count::current_output_count().unwrap_or(1);
    if count > 4 || args.is_empty() {
        return Err(invalid(NAME, "expected a sample and at most four outputs"));
    }
    // No allocator certificate is claimed for lane scratch and result storage.
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME,
        "finite host-memory execution is unsupported; inferential-statistics lane and result workspaces have no complete resource bound")?;
    inferential::admit_automatic_host_arguments(NAME, args.view())?;
    let _sources = args.fork();
    inferential::real_sample(NAME, &args[0])?;
    let second = args.get(1).filter(|value| {
        crate::builtins::common::validation::value_text_scalar_string(value).is_none()
    });
    if let Some(value) = second {
        inferential::real_sample(NAME, value)?;
    }
    let options =
        options::Options::parse(args.view(), if second.is_some() { 2 } else { 1 }, runtime).await?;
    let mut cooperation = Cooperation::new(runtime);
    cooperation.completed(0).await?;
    let x = inferential::gather(NAME, &args[0], runtime).await?;
    let y = match second {
        Some(value) => Some(inferential::gather(NAME, value, runtime).await?),
        None => None,
    };
    let mut outputs = engine::evaluate(&x, y.as_ref(), options, &mut cooperation).await?;
    cooperation.completed(0).await?;
    outputs.truncate(count);
    Ok(Value::output_list(outputs))
}
