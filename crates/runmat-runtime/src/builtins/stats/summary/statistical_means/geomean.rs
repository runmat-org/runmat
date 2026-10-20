use super::*;
use options::ReductionKind;
use runmat_builtins::{GEOMEAN_INTEGER_CONTROL_EXTENSION, GEOMEAN_INTEGER_DATA_EXTENSION};

#[runmat_macros::runtime_builtin(
    name = "geomean",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::statistical_means::geomean"
)]
pub(crate) async fn geomean_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let kind = ReductionKind::Geometric;
    admit_data(kind, &value)?;
    if data::integer_data(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GEOMEAN_INTEGER_DATA_EXTENSION,
            kind.name(),
        )?;
    }
    let options = options::parse(kind, &rest).await?;
    if options.typed_integer_control {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GEOMEAN_INTEGER_CONTROL_EXTENSION,
            kind.name(),
        )?;
    }
    if let Value::GpuTensor(handle) = &value {
        if let Some(result) = resident::try_reduce(kind, handle, &options).await? {
            return Ok(result);
        }
    }
    let (values, shape, dtype) = data::materialize(kind, value).await?.real_values(kind)?;
    host::reduce(kind, values, shape, dtype, &options)
}
#[cfg(test)]
pub(crate) async fn geomean_builtin_source(
    value: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    geomean_builtin(
        value,
        crate::call::BuiltinArguments::from_borrowed(&rest, None)?,
    )
    .await
}
