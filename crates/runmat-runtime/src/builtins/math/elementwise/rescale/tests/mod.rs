mod broadcast;
mod errors;
mod provider;
mod ranges;
mod support;
mod types;

pub(super) async fn rescale_builtin(
    value: runmat_value::Value,
    rest: Vec<runmat_value::Value>,
) -> crate::BuiltinResult<runmat_value::Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::rescale_builtin_registered(value, rest).await
}
