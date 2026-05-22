//! MATLAB-compatible `clearvars` builtin for workspace variables.

use std::collections::HashSet;

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, workspace, BuiltinResult};

#[runtime_builtin(
    name = "clearvars",
    category = "introspection",
    summary = "Clear variables from the active workspace, with optional exclusions.",
    keywords = "clearvars,workspace,variables,except",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::introspection::clearvars"
)]
async fn clearvars_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        workspace::clear().map_err(clearvars_error)?;
        return Ok(empty_return_value());
    }

    let mut words = Vec::new();
    for arg in &args {
        collect_clearvars_words(arg, &mut words)?;
    }

    let mut targets = Vec::new();
    let mut exclusions = Vec::new();
    let mut saw_except = false;
    for word in words.iter() {
        let name = word.trim();
        if name.is_empty() {
            continue;
        }
        if name.eq_ignore_ascii_case("-except") {
            if saw_except {
                return Err(clearvars_error("clearvars: duplicate -except option"));
            }
            saw_except = true;
            continue;
        }
        if name.starts_with('-') {
            return Err(clearvars_error(format!(
                "clearvars: unsupported option '{name}'"
            )));
        }
        if saw_except {
            exclusions.push(name.to_string());
        } else {
            targets.push(name.to_string());
        }
    }

    if saw_except {
        if exclusions.is_empty() {
            return Err(clearvars_error(
                "clearvars: -except requires at least one variable name",
            ));
        }
        clear_except(&targets, &exclusions)?;
    } else {
        for name in targets {
            workspace::remove(&name).map_err(clearvars_error)?;
        }
    }

    Ok(empty_return_value())
}

fn clear_except(targets: &[String], exclusions: &[String]) -> BuiltinResult<()> {
    let keep: HashSet<&str> = exclusions.iter().map(String::as_str).collect();
    if targets.is_empty() {
        let snapshot = workspace::snapshot()
            .ok_or_else(|| clearvars_error("clearvars: workspace state unavailable"))?;
        for (name, _) in snapshot {
            if !keep.contains(name.as_str()) {
                workspace::remove(&name).map_err(clearvars_error)?;
            }
        }
    } else {
        for name in targets {
            if !keep.contains(name.as_str()) {
                workspace::remove(name).map_err(clearvars_error)?;
            }
        }
    }
    Ok(())
}

fn clearvars_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("clearvars")
        .build()
}

fn collect_clearvars_words(arg: &Value, out: &mut Vec<String>) -> BuiltinResult<()> {
    match arg {
        Value::String(text) => {
            out.push(text.trim().to_string());
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows > 1 {
                return Err(clearvars_error(
                    "clearvars: character array inputs must be a row vector or scalar text value",
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
        _ => Err(clearvars_error(
            "clearvars: expected variable names as character vectors or string scalars",
        )),
    }
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}
