mod host;
mod provider;
mod representations;
mod target;

pub(super) async fn typecast_builtin(
    args: Vec<runmat_value::Value>,
) -> crate::BuiltinResult<runmat_value::Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    super::typecast_builtin_registered(args).await
}
