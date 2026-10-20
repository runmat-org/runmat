use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "random",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::random"
)]
pub(crate) async fn random_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    distribution: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::fitted_distribution::sample_distribution(distribution, rest, runtime.as_ref()).await
}

#[cfg(test)]
pub(crate) async fn random_builtin(distribution: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::fitted_distribution::sample_distribution(
        distribution,
        crate::builtins::common::test_support::builtin_arguments(rest),
        None,
    )
    .await
}
