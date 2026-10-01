use super::*;
use options::ReductionKind;
use runmat_builtins::{RMSE_INTEGER_DATA_EXTENSION, RMSE_INTEGER_WEIGHTS_EXTENSION};

#[runmat_macros::runtime_builtin(
    name = "rmse",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::statistical_means::rmse"
)]
pub(crate) async fn rmse_builtin(
    forecast: Value,
    actual: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let kind = ReductionKind::RootMeanSquaredError;
    admit_data(kind, &forecast)?;
    admit_data(kind, &actual)?;
    if data::integer_data(&forecast) || data::integer_data(&actual) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RMSE_INTEGER_DATA_EXTENSION,
            kind.name(),
        )?;
    }
    let options = options::parse(kind, rest).await?;
    if options
        .weights
        .as_ref()
        .is_some_and(data::exact_integer_weight)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RMSE_INTEGER_WEIGHTS_EXTENSION,
            kind.name(),
        )?;
    }
    if let Some(weights) = options.weights.as_ref() {
        admit_data(kind, weights)?;
    }
    let forecast = data::materialize(kind, forecast).await?;
    let actual = data::materialize(kind, actual).await?;
    let (values, shape, dtype, forecast_shape, actual_shape) =
        host::residuals(forecast, actual, kind)?;
    if let Some(weights) = options.weights.clone() {
        let weights = data::materialize(kind, weights).await?;
        host::weighted_rmse(
            values,
            shape,
            &forecast_shape,
            &actual_shape,
            dtype,
            &options,
            weights,
        )
    } else {
        host::reduce(kind, values, shape, dtype, &options)
    }
}
