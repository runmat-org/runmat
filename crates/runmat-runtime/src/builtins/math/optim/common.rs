use runmat_builtins::{CharArray, StructValue, Tensor, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) fn optim_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn canonicalize_callback_handle(handle: &Value) -> Value {
    fn resolve_text_handle(text: &str) -> Option<Value> {
        let name = text.strip_prefix('@')?;
        if name.is_empty() {
            return None;
        }
        let function = crate::user_functions::resolve_semantic_function_by_name(name)?;
        Some(Value::SemanticFunctionHandle {
            name: name.to_string(),
            function,
        })
    }

    match handle {
        Value::String(text) => resolve_text_handle(text).unwrap_or_else(|| handle.clone()),
        Value::StringArray(array) if array.data.len() == 1 => {
            resolve_text_handle(&array.data[0]).unwrap_or_else(|| handle.clone())
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            resolve_text_handle(&text).unwrap_or_else(|| handle.clone())
        }
        Value::FunctionHandle(name) => {
            if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(name) {
                Value::SemanticFunctionHandle {
                    name: name.clone(),
                    function,
                }
            } else {
                handle.clone()
            }
        }
        Value::ExternalFunctionHandle(name) => {
            if crate::is_well_formed_qualified_name(name) {
                if let Some(function) =
                    crate::user_functions::resolve_semantic_function_by_name(name)
                {
                    return Value::SemanticFunctionHandle {
                        name: name.clone(),
                        function,
                    };
                }
            }
            handle.clone()
        }
        Value::Closure(closure) => {
            if closure.semantic_function.is_none() {
                if let Some(function) =
                    crate::user_functions::resolve_semantic_function_by_name(&closure.function_name)
                {
                    let mut bound = closure.clone();
                    bound.semantic_function = Some(function);
                    return Value::Closure(bound);
                }
            }
            handle.clone()
        }
        _ => handle.clone(),
    }
}

pub(crate) async fn call_function(handle: &Value, args: Vec<Value>) -> BuiltinResult<Value> {
    let callback = canonicalize_callback_handle(handle);
    crate::call_feval_async_with_outputs(callback, &args, 1).await
}

pub(crate) async fn call_scalar_function(name: &str, handle: &Value, x: f64) -> BuiltinResult<f64> {
    let value = call_function(handle, vec![Value::Num(x)]).await?;
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    value_to_scalar(name, value)
}

pub(crate) fn value_to_scalar(name: &str, value: Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => ensure_finite(name, n),
        Value::Int(i) => ensure_finite(name, i.to_f64()),
        Value::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                ensure_finite(name, tensor.data[0])
            } else {
                Err(optim_error(
                    name,
                    format!("{name}: function value must be a scalar"),
                ))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
            } else {
                Err(optim_error(
                    name,
                    format!("{name}: function value must be a scalar"),
                ))
            }
        }
        other => Err(optim_error(
            name,
            format!("{name}: function value must be real numeric, got {other:?}"),
        )),
    }
}

pub(crate) async fn value_to_real_vector(name: &str, value: Value) -> BuiltinResult<Vec<f64>> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) => Ok(vec![ensure_finite(name, n)?]),
        Value::Int(i) => Ok(vec![ensure_finite(name, i.to_f64())?]),
        Value::Bool(b) => Ok(vec![if b { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => finite_vec(name, tensor.data),
        Value::LogicalArray(logical) => Ok(logical
            .data
            .iter()
            .map(|&v| if v != 0 { 1.0 } else { 0.0 })
            .collect()),
        other => Err(optim_error(
            name,
            format!("{name}: function value must be a real numeric vector, got {other:?}"),
        )),
    }
}

pub(crate) async fn initial_guess(name: &str, value: Value) -> BuiltinResult<InitialGuess> {
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) => Ok(InitialGuess {
            values: vec![ensure_finite(name, n)?],
            shape: vec![1, 1],
            scalar: true,
        }),
        Value::Int(i) => Ok(InitialGuess {
            values: vec![ensure_finite(name, i.to_f64())?],
            shape: vec![1, 1],
            scalar: true,
        }),
        Value::Bool(b) => Ok(InitialGuess {
            values: vec![if b { 1.0 } else { 0.0 }],
            shape: vec![1, 1],
            scalar: true,
        }),
        Value::Tensor(tensor) => {
            if tensor.data.is_empty() {
                return Err(optim_error(
                    name,
                    format!("{name}: initial guess cannot be empty"),
                ));
            }
            Ok(InitialGuess {
                values: finite_vec(name, tensor.data)?,
                shape: tensor.shape,
                scalar: false,
            })
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Err(optim_error(
                    name,
                    format!("{name}: initial guess cannot be empty"),
                ));
            }
            Ok(InitialGuess {
                values: logical
                    .data
                    .iter()
                    .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                    .collect(),
                shape: logical.shape,
                scalar: false,
            })
        }
        other => Err(optim_error(
            name,
            format!("{name}: initial guess must be real numeric, got {other:?}"),
        )),
    }
}

pub(crate) fn vector_to_value(
    name: &str,
    values: Vec<f64>,
    shape: &[usize],
    scalar: bool,
) -> BuiltinResult<Value> {
    if scalar {
        Ok(Value::Num(values[0]))
    } else {
        Tensor::new(values, shape.to_vec())
            .map(Value::Tensor)
            .map_err(|e| optim_error(name, format!("{name}: {e}")))
    }
}

pub(crate) fn field_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(CharArray { data, rows: 1, .. }) => Ok(data.iter().collect()),
        other => Err(optim_error(
            "optimset",
            format!("optimset: option names must be strings, got {other:?}"),
        )),
    }
}

pub(crate) fn lookup_option<'a>(options: &'a StructValue, name: &str) -> Option<&'a Value> {
    options
        .fields
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value)
}

pub(crate) fn option_f64(
    builtin: &str,
    options: Option<&StructValue>,
    field: &str,
    default: f64,
) -> BuiltinResult<f64> {
    let Some(options) = options else {
        return Ok(default);
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(default);
    };
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        other => {
            return Err(optim_error(
                builtin,
                format!("{builtin}: option {field} must be numeric, got {other:?}"),
            ))
        }
    };
    ensure_finite(builtin, parsed)
}

pub(crate) fn option_usize(
    builtin: &str,
    options: Option<&StructValue>,
    field: &str,
    default: usize,
) -> BuiltinResult<usize> {
    let value = option_f64(builtin, options, field, default as f64)?;
    if value < 0.0 {
        return Err(optim_error(
            builtin,
            format!("{builtin}: option {field} must be non-negative"),
        ));
    }
    Ok(value.floor() as usize)
}

pub(crate) fn option_string(
    options: Option<&StructValue>,
    field: &str,
    default: &str,
) -> BuiltinResult<String> {
    let Some(options) = options else {
        return Ok(default.to_string());
    };
    let Some(value) = lookup_option(options, field) else {
        return Ok(default.to_string());
    };
    match value {
        Value::String(s) => Ok(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(CharArray { data, rows: 1, .. }) => {
            Ok(data.iter().collect::<String>().to_ascii_lowercase())
        }
        other => Err(optim_error(
            "optim",
            format!("optim option {field} must be a string, got {other:?}"),
        )),
    }
}

fn ensure_finite(name: &str, value: f64) -> BuiltinResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(optim_error(
            name,
            format!("{name}: function value must be finite"),
        ))
    }
}

fn finite_vec(name: &str, values: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(values)
    } else {
        Err(optim_error(
            name,
            format!("{name}: function value must be finite"),
        ))
    }
}

pub(crate) struct InitialGuess {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
    pub scalar: bool,
}

#[cfg(test)]
mod tests {
    use super::canonicalize_callback_handle;
    use runmat_builtins::{CharArray, Closure, StringArray, Value};
    use std::sync::Arc;

    #[test]
    fn callback_handle_canonicalizer_binds_function_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(42)
            })));
        let canonical = canonicalize_callback_handle(&Value::FunctionHandle("decay".to_string()));
        assert_eq!(
            canonical,
            Value::SemanticFunctionHandle {
                name: "decay".to_string(),
                function: 42,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_qualified_external_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(43)
            })));
        let canonical =
            canonicalize_callback_handle(&Value::ExternalFunctionHandle("pkg.decay".to_string()));
        assert_eq!(
            canonical,
            Value::SemanticFunctionHandle {
                name: "pkg.decay".to_string(),
                function: 43,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_keeps_malformed_external_handle_name_shaped() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg..decay").then_some(44)
            })));
        let raw = Value::ExternalFunctionHandle("pkg..decay".to_string());
        let canonical = canonicalize_callback_handle(&raw);
        assert_eq!(canonical, raw);
    }

    #[test]
    fn callback_handle_canonicalizer_binds_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(45)
            })));
        let canonical = canonicalize_callback_handle(&Value::String("@decay".to_string()));
        assert_eq!(
            canonical,
            Value::SemanticFunctionHandle {
                name: "decay".to_string(),
                function: 45,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_string_array_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "pkg.decay").then_some(46)
            })));
        let canonical = canonicalize_callback_handle(&Value::StringArray(
            StringArray::new(vec!["@pkg.decay".to_string()], vec![1, 1]).expect("string array"),
        ));
        assert_eq!(
            canonical,
            Value::SemanticFunctionHandle {
                name: "pkg.decay".to_string(),
                function: 46,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_char_text_handle_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(47)
            })));
        let canonical =
            canonicalize_callback_handle(&Value::CharArray(CharArray::new_row("@decay")));
        assert_eq!(
            canonical,
            Value::SemanticFunctionHandle {
                name: "decay".to_string(),
                function: 47,
            }
        );
    }

    #[test]
    fn callback_handle_canonicalizer_binds_name_only_closure_when_resolved() {
        let _resolver_guard =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "decay").then_some(48)
            })));
        let raw = Value::Closure(Closure {
            function_name: "decay".to_string(),
            semantic_function: None,
            captures: vec![Value::Num(9.0)],
        });
        let canonical = canonicalize_callback_handle(&raw);
        assert_eq!(
            canonical,
            Value::Closure(Closure {
                function_name: "decay".to_string(),
                semantic_function: Some(48),
                captures: vec![Value::Num(9.0)],
            })
        );
    }

    #[test]
    fn callback_handle_canonicalizer_keeps_name_only_closure_without_resolver() {
        let raw = Value::Closure(Closure {
            function_name: "decay".to_string(),
            semantic_function: None,
            captures: vec![Value::Num(9.0)],
        });
        let canonical = canonicalize_callback_handle(&raw);
        assert_eq!(canonical, raw);
    }
}
