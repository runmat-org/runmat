mod contents;
mod error;
mod input;
mod output;
pub(crate) mod specification;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "what",
    builtin_path = "crate::builtins::io::repl_fs::source_inventory::what"
)]
async fn what_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let folder = input::folder(&args)?;
    let inventory = contents::inspect(folder).await?;
    output::value(inventory)
}

#[cfg(test)]
mod tests;
