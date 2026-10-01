use super::*;

#[runtime_builtin(
    name = "prctile",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::order_stats::prctile"
)]
pub(crate) async fn prctile_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::ensure_quantile_integer_extensions(
        "prctile",
        &value,
        &rest,
        &runmat_builtins::PRCTILE_DATA_EXTENSION,
        &runmat_builtins::PRCTILE_PROBABILITY_EXTENSION,
    )?;
    let args = super::parse_quantile_args("prctile", value, rest, QuantileRequestUnit::Percentage).await?;
    super::quantile_tensor(args, "prctile")
}
