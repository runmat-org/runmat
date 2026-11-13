mod error;
mod input;
mod output;
mod query;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "ls",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::directory_listing::ls"
)]
async fn ls_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::LS_DESCRIPTOR,
        &runmat_builtins::LS_ERROR_TOO_MANY_OUTPUTS,
        "ls",
    )?;
    let input = input::Input::parse(args.view())?;
    let rows = query::execute(input).await?;
    if output::should_emit_stdout() {
        output::emit(&rows)?;
    }
    output::value(&rows)
}

#[cfg(test)]
mod tests;
