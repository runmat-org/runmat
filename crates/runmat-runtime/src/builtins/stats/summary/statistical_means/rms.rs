use super::*;
use options::ReductionKind;
use runmat_builtins::RMS_INTEGER_DATA_EXTENSION;

#[runmat_macros::runtime_builtin(
    name = "rms",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::statistical_means::rms"
)]
pub(crate) async fn rms_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let kind = ReductionKind::RootMeanSquare;
    admit_data(kind, &value)?;
    if data::integer_data(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RMS_INTEGER_DATA_EXTENSION,
            kind.name(),
        )?;
    }
    let options = options::parse(kind, &rest).await?;
    if let Value::GpuTensor(handle) = &value {
        if let Some(result) = resident::try_reduce(kind, handle, &options).await? {
            return Ok(result);
        }
    }
    let (values, shape, dtype) = data::materialize(kind, value).await?.scaled_magnitudes();
    host::reduce_rms(kind, values, shape, dtype, &options)
}
#[cfg(test)]
pub(crate) async fn rms_builtin_source(
    value: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    rms_builtin(
        value,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
