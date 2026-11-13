use runmat_builtins::{ISFOLDER_ERROR_ARITY, ISFOLDER_ERROR_PATH};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isfolder",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::path_predicate::isfolder"
)]
async fn isfolder_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ISFOLDER_DESCRIPTOR,
        &runmat_builtins::ISFOLDER_ERROR_TOO_MANY_OUTPUTS,
        "isfolder",
    )?;
    super::execute::evaluate(
        args.view(),
        "isfolder",
        &ISFOLDER_ERROR_ARITY,
        &ISFOLDER_ERROR_PATH,
        runmat_filesystem::FsMetadata::is_dir,
    )
    .await
}
