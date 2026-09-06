pub(crate) mod specification;
#[cfg(test)]
mod tests;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

#[runtime_builtin(
    name = "pathsep",
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::pathsep"
)]
async fn pathsep_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
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
