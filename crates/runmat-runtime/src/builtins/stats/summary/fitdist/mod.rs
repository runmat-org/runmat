use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "fitdist",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::fitdist"
)]
pub(crate) async fn fitdist_builtin(
    data: Value,
    distname: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::fitted_distribution::fit_sample_distribution(data, distname, rest).await
}
