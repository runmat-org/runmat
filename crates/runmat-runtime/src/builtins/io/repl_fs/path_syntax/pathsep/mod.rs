#[cfg(test)]
mod tests;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

#[runtime_builtin(
    name = "pathsep",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::pathsep"
)]
async fn pathsep_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PATHSEP_DESCRIPTOR,
        &runmat_builtins::PATHSEP_ERROR_TOO_MANY_OUTPUTS,
        "pathsep",
    )?;
    if !args.is_empty() {
        return Err(super::error::catalog(
            &runmat_builtins::PATHSEP_ERROR_ARITY,
            "pathsep",
        ));
    }
    Ok(Value::CharArray(CharArray::new_row(
        &crate::builtins::common::path_state::PATH_LIST_SEPARATOR.to_string(),
    )))
}
