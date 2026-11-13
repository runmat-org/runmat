use runmat_builtins::{ISFILE_ERROR_ARITY, ISFILE_ERROR_PATH};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isfile",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::path_predicate::isfile"
)]
async fn isfile_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ISFILE_DESCRIPTOR,
        &runmat_builtins::ISFILE_ERROR_TOO_MANY_OUTPUTS,
        "isfile",
    )?;
    super::execute::evaluate(
        args.view(),
        "isfile",
        &ISFILE_ERROR_ARITY,
        &ISFILE_ERROR_PATH,
        runmat_filesystem::FsMetadata::is_file,
    )
    .await
}
