pub(super) mod execute;
mod filesystem;
mod plan;
mod result;
mod single;
mod wildcard;

use super::authority::TransferFilesystem;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "copyfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::file_transfer::copyfile"
)]
async fn copyfile_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let outcome = execute::evaluate_retained(args, context.as_ref()).await?;
    if let Some(count) = crate::output_count::current_output_count() {
        if count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            count,
            outcome.outputs(),
        ));
    }
    Ok(outcome.first_output())
}
