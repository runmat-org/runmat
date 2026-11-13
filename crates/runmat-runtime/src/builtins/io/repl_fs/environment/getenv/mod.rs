mod execute;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
#[runtime_builtin(
    name = "getenv",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::environment::getenv"
)]
async fn getenv_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::builtins::common::env::require_read()?;
    execute::run(runtime.as_ref(), args)
}
