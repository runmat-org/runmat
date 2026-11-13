mod contents;
mod error;
mod input;
mod output;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "what",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::source_inventory::what"
)]
async fn what_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::WHAT_DESCRIPTOR,
        &runmat_builtins::WHAT_ERROR_TOO_MANY_OUTPUTS,
        "what",
    )?;
    let folder = input::folder(args.view())?;
    let inventory = contents::inspect(folder).await?;
    output::value(inventory)
}

#[cfg(test)]
mod tests;
