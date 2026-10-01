use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "random",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::random"
)]
pub(crate) async fn random_builtin(distribution: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::fitted_distribution::sample_distribution(distribution, rest).await
}
