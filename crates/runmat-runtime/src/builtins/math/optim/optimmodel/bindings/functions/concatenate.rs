//! Concatenation uses the actual canonical numeric dimension decoder.
use super::super::*;
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.cat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::concatenate"
)]
async fn cat(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_CAT_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    let first = args
        .view()
        .first()
        .ok_or_else(|| policy.invalid("cat requires a dimension"))?;
    if !matches!(first, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
        return Err(policy.invalid("cat dimension must be a numeric scalar"));
    }
    if let Value::Tensor(array) = first {
        array
            .validate_shape()
            .map_err(|source| policy.invalid(source))?;
    }
    let dimension = crate::builtins::common::tensor::parse_dimension(first, "cat")
        .map_err(|source| policy.invalid(source))?;
    let dimension = dimension
        .checked_sub(1)
        .ok_or_else(|| policy.invalid("cat dimension must be positive"))?;
    let values = args.view().range(1..)?;
    super::super::super::functions::concatenate(dimension, values, policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.horzcat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::concatenate"
)]
async fn horzcat(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_HORZCAT_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    let dimension = 1;
    let values = args.view();
    super::super::super::functions::concatenate(dimension, values, policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.vertcat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::concatenate"
)]
async fn vertcat(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_VERTCAT_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    let dimension = 0;
    let values = args.view();
    super::super::super::functions::concatenate(dimension, values, policy, &mut cooperation).await
}
