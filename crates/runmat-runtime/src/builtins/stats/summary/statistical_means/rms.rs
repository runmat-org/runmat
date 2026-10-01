use super::*;
use options::ReductionKind;
use runmat_builtins::RMS_INTEGER_DATA_EXTENSION;

#[runmat_macros::runtime_builtin(
    name = "rms",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::statistical_means::rms"
)]
pub(crate) async fn rms_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let kind = ReductionKind::RootMeanSquare;
    admit_data(kind, &value)?;
    if data::integer_data(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RMS_INTEGER_DATA_EXTENSION,
            kind.name(),
        )?;
    }
    let options = options::parse(kind, rest).await?;
    let (values, shape, dtype) = data::materialize(kind, value).await?.magnitudes();
    host::reduce(kind, values, shape, dtype, &options)
}
