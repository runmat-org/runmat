mod error;
mod input;
mod output;
mod query;
pub(crate) mod specification;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "ls",
    category = "io/repl_fs",
    summary = "List files and folders in directories or wildcard paths.",
    keywords = "ls,list files,folder contents,wildcard listing,dir",
    accel = "cpu",
    suppress_auto_output = true,
    descriptor(runmat_builtins::LS_DESCRIPTOR),
    integer_audit(runmat_builtins::LS_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::directory_listing::ls"
)]
async fn ls_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let input = input::Input::parse(&args)?;
    let rows = query::execute(input).await?;
    if output::should_emit_stdout() {
        output::emit(&rows);
    }
    output::value(&rows)
}

#[cfg(test)]
mod tests;
