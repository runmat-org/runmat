use runmat_builtins::{ISFOLDER_ERROR_ARITY, ISFOLDER_ERROR_PATH};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isfolder",
    builtin_path = "crate::builtins::io::repl_fs::path_predicate::isfolder"
)]
async fn isfolder_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    super::execute::evaluate(
        &args,
        "isfolder",
        &ISFOLDER_ERROR_ARITY,
        &ISFOLDER_ERROR_PATH,
        runmat_filesystem::FsMetadata::is_dir,
    )
    .await
}
