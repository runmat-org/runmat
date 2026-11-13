mod error;
mod input;
mod output;
mod query;
mod record;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "dir",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::directory_listing::dir"
)]
async fn dir_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let input = input::Input::parse(args.view())?;
    let records = query::execute(input).await?;
    if output::should_emit_stdout() {
        output::emit(&records)?;
    }
    output::value(records)
}

#[cfg(test)]
mod tests;
