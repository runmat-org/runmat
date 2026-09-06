pub(super) mod execute;
mod filesystem;
mod plan;
mod result;
mod single;
pub(crate) mod specification;
mod wildcard;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "copyfile",
    builtin_path = "crate::builtins::io::repl_fs::file_transfer::copyfile"
)]
async fn copyfile_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let outcome = execute::evaluate(&args).await?;
    if let Some(count) = crate::output_count::current_output_count() {
        if count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            count,
            outcome.outputs(),
        ));
    }
    Ok(outcome.first_output())
}
