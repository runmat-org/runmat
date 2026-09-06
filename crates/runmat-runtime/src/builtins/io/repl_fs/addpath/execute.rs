use runmat_value::Value;

use super::super::path_mutation::text::NumericPolicy;

pub(super) async fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(super::errors::descriptor(
            &runmat_builtins::ADDPATH_ERROR_TOO_FEW_ARGUMENTS,
        ));
    }
    let tokens = super::super::path_mutation::text::decode(
        args,
        super::errors::NAME,
        NumericPolicy::RunMatExtension(&runmat_builtins::ADDPATH_NUMERIC_CHARACTER_CODES_EXTENSION),
    )
    .await
    .map_err(super::errors::decode)?;
    let request = super::arguments::parse(tokens)?;
    let replacement = super::operation::plan(request).await?;
    let previous = crate::builtins::common::path_state::current_path_string();
    crate::builtins::common::path_state::set_path_string(&replacement);
    Ok(Value::CharArray(runmat_value::CharArray::new_row(
        &previous,
    )))
}
