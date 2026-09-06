use runmat_value::Value;

pub(super) async fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(super::errors::descriptor(
            &runmat_builtins::RMPATH_ERROR_TOO_FEW_ARGUMENTS,
        ));
    }
    let tokens = super::super::path_mutation::text::decode(
        args,
        super::errors::NAME,
        super::super::path_mutation::text::NumericPolicy::Reject,
    )
    .await
    .map_err(super::errors::decode)?;
    let directories = super::arguments::parse(tokens)?;
    let replacement = super::operation::plan(directories).await?;
    let previous = crate::builtins::common::path_state::current_path_string();
    crate::builtins::common::path_state::set_path_string(&replacement);
    Ok(Value::CharArray(runmat_value::CharArray::new_row(
        &previous,
    )))
}
