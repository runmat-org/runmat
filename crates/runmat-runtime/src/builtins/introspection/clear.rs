//! MATLAB-compatible `clear` builtin for workspace variables.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, workspace, BuiltinResult};

#[runtime_builtin(
    name = "clear",
    category = "introspection",
    summary = "Clear variables from the active workspace.",
    keywords = "clear,workspace,variables",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::introspection::clear"
)]
async fn clear_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        workspace::clear().map_err(clear_error)?;
        return Ok(empty_return_value());
    }

    let mut names = Vec::new();
    for arg in &args {
        collect_clear_targets(arg, &mut names)?;
    }

    if names.is_empty() || names.iter().any(|name| name.eq_ignore_ascii_case("all")) {
        workspace::clear().map_err(clear_error)?;
    } else {
        for name in names {
            workspace::remove(&name).map_err(clear_error)?;
        }
    }

    Ok(empty_return_value())
}

fn clear_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("clear").build()
}

fn collect_clear_targets(arg: &Value, out: &mut Vec<String>) -> BuiltinResult<()> {
    match arg {
        Value::String(text) => {
            out.push(text.trim().to_string());
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows > 1 {
                return Err(clear_error(
                    "clear: character array inputs must be a row vector or scalar text value",
                ));
            }
            out.push(chars.data.iter().collect::<String>().trim().to_string());
            Ok(())
        }
        Value::StringArray(array) => {
            for text in &array.data {
                out.push(text.trim().to_string());
            }
            Ok(())
        }
        _ => Err(clear_error(
            "clear: expected variable names as character vectors or string scalars",
        )),
    }
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}
