mod error;
mod input;
mod output;
mod query;
mod record;
pub(crate) mod specification;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "dir",
    category = "io/repl_fs",
    summary = "Return file and folder metadata.",
    keywords = "dir,list files,folder contents,metadata,wildcard,struct array",
    accel = "cpu",
    suppress_auto_output = true,
    descriptor(runmat_builtins::DIR_DESCRIPTOR),
    integer_audit(runmat_builtins::DIR_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::directory_listing::dir"
)]
async fn dir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let input = input::Input::parse(&args)?;
    let records = query::execute(input).await?;
    if output::should_emit_stdout() {
        output::emit(&records);
    }
    output::value(records)
}

#[cfg(test)]
mod tests;
