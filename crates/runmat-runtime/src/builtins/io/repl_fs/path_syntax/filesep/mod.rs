pub(crate) mod specification;
#[cfg(test)]
mod tests;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

#[runtime_builtin(
    name = "filesep",
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::filesep"
)]
async fn filesep_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(super::error::catalog(
            &runmat_builtins::FILESEP_ERROR_ARITY,
            "filesep",
        ));
    }
    Ok(Value::CharArray(CharArray::new_row(
        std::path::MAIN_SEPARATOR_STR,
    )))
}
