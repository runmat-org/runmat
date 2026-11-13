use super::*;

#[runtime_builtin(
    name = "quantile",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::order_stats::quantile"
)]
pub(crate) async fn quantile_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::ensure_quantile_integer_extensions(
        "quantile",
        &value,
        rest.view(),
        &runmat_builtins::QUANTILE_DATA_EXTENSION,
        &runmat_builtins::QUANTILE_PROBABILITY_EXTENSION,
    )?;
    let args =
        super::parse_quantile_args("quantile", value, rest, QuantileRequestUnit::Probability)
            .await?;
    super::quantile_tensor(args, "quantile")
}
