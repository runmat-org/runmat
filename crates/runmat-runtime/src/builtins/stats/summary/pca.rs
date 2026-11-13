//! Principal components of real observations, through the canonical SVD owner.
use super::inferential::{self, invalid, Cooperation};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};
mod engine;
mod options;
mod prepare;
#[cfg(test)]
mod tests;
const NAME: &str = "pca";

#[runtime_builtin(
    name = "pca",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::pca"
)]
async fn pca_builtin(
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
    if args.is_empty() || count > 6 {
        return Err(invalid(NAME, "expected X and at most six outputs"));
    }
    crate::context::host_allocation::require_unbounded_workspace(runtime,NAME,
        "finite host-memory execution is unsupported; PCA SVD and result workspaces have no complete allocation bound")?;
    inferential::admit_automatic_host_arguments(NAME, args.view())?;
    inferential::real_sample(NAME, &args[0])?;
    let _sources = args.fork();
    let options = options::Options::parse(args.view(), runtime).await?;
    let mut cooperation = Cooperation::new(runtime);
    cooperation.completed(0).await?;
    let input = inferential::gather(NAME, &args[0], runtime).await?;
    let prepared = prepare::prepare(&input, &options, &mut cooperation).await?;
    let outputs = engine::evaluate(prepared, &options, count, runtime, &mut cooperation).await?;
    cooperation.completed(0).await?;
    Ok(Value::output_list(outputs))
}

fn output(values: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> BuiltinResult<Value> {
    Tensor::new_with_dtype(values, shape, dtype)
        .map(Value::Tensor)
        .map_err(|error| invalid(NAME, error))
}
