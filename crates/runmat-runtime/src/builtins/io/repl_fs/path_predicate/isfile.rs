use runmat_builtins::{ISFILE_ERROR_ARITY, ISFILE_ERROR_PATH};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isfile",
    builtin_path = "crate::builtins::io::repl_fs::path_predicate::isfile"
)]
async fn isfile_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    super::execute::evaluate(
        &args,
        "isfile",
        &ISFILE_ERROR_ARITY,
        &ISFILE_ERROR_PATH,
        runmat_filesystem::FsMetadata::is_file,
    )
    .await
}
