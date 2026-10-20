mod execute;
mod input;
#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "fullfile",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::path_syntax::fullfile"
)]
async fn fullfile_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    execute::run(args).await
}
