use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "cdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::cdf"
)]
pub(crate) async fn cdf_builtin(
    distribution: Value,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::fitted_distribution::evaluate_cdf_entry(distribution, x, rest).await
}
