//! Invocation-owned Nelder-Mead direct search, without derivative assumptions.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    FMINSEARCH_CATALOG_ENTRY, FMINSEARCH_DESCRIPTOR, FMINSEARCH_ERROR_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
mod input;
pub(in crate::builtins::math::optim) mod options;
mod evaluation;
pub(in crate::builtins::math::optim) mod simplex;
pub(in crate::builtins::math::optim) mod observers;
mod reporting;
#[cfg(test)]
mod tests;
const NAME: &str = "fminsearch";

fn invalid(message: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("fminsearch: {message}"))
        .with_builtin(NAME)
        .with_identifier("RunMat:fminsearch:InvalidArgument")
        .build()
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("fminsearch: workspace geometry or allocation failed")
        .with_builtin(NAME)
        .with_identifier("RunMat:fminsearch:AllocationFailed")
        .with_source(source)
        .build()
}
fn buffer<T>(count: usize) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(count).map_err(allocation)
}

#[runtime_builtin(
    name = "fminsearch",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fminsearch"
)]
async fn fminsearch_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(runtime.as_ref(), args).await
}
async fn evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME,
        "finite host-memory execution is unsupported; simplex, callback and result workspaces have no complete resource bound")?;
    crate::output_count::validate_bounded_outputs(
        &FMINSEARCH_DESCRIPTOR,
        &FMINSEARCH_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let _sources = args.fork();
    let input = input::Input::parse(args.view())?;
    input::admit_numeric(input.x0)?;
    options::admit_in(input.options, runtime)?;
    let mut cooperation =
        ScalarCooperation::for_builtin(FMINSEARCH_CATALOG_ENTRY.identity, runtime);
    cooperation.completed(0).await?;
    let guess = input::initial(input.x0, runtime, &mut cooperation).await?;
    let options = options::Options::parse(input.options, guess.values.len(), runtime).await?;
    let mut observers = observers::Observers::new(&options, &guess.shape, guess.scalar);
    let evaluator = evaluation::Evaluator::new(
        input.function,
        &guess.shape,
        guess.scalar,
        &options,
        &mut cooperation,
    );
    let result = simplex::minimize(guess.values, evaluator, &options, &mut observers).await?;
    let count = crate::output_count::current_output_count();
    if count == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    let x = common::vector_to_value(NAME, result.point.x, &guess.shape, guess.scalar)?;
    let Some(count) = count else { return Ok(x) };
    let mut outputs = Vec::with_capacity(count);
    outputs.push(x);
    if count >= 2 {
        outputs.push(Value::Num(result.point.f));
    }
    if count >= 3 {
        outputs.push(Value::Num(result.reason.flag()));
    }
    if count >= 4 {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("iterations", Value::Num(result.iterations as f64));
        fields.insert("funcCount", Value::Num(result.func_count as f64));
        fields.insert(
            "algorithm",
            Value::from("Nelder-Mead simplex direct search"),
        );
        fields.insert("message", Value::from(result.message));
        outputs.push(Value::Struct(fields.finish()));
    }
    Ok(Value::output_list(outputs))
}
