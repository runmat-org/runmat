use crate::{user_functions, BuiltinResult};
use runmat_value::{Closure, Value};

use super::super::error::arrayfun_flow;
use super::Callable;

impl Callable {
    fn resolved_semantic_handle(name: &str) -> Option<Self> {
        let function = user_functions::resolve_semantic_function_by_name(name)?;
        Some(Self::Closure(Closure {
            function_name: name.into(),
            bound_function: Some(function),
            captures: Vec::new(),
        }))
    }

    fn resolved_builtin(name: &str) -> Option<Self> {
        runmat_builtins::builtin_catalog_entry_by_name(name).map(|entry| Self::Builtin {
            identity: entry.identity,
        })
    }

    pub(in crate::builtins::acceleration::gpu::arrayfun) fn from_function(
        value: Value,
    ) -> BuiltinResult<Self> {
        match value {
            Value::String(text) => Self::from_text(&text),
            Value::CharArray(value) if value.rows == 1 => {
                Self::from_text(&value.data.iter().collect::<String>())
            }
            Value::StringArray(value) if value.data.len() == 1 => Self::from_text(&value.data[0]),
            Value::FunctionHandle(name) => Self::from_text(&name),
            Value::ExternalFunctionHandle(name) => Self::from_external_name(name),
            Value::BoundFunctionHandle { name, function } => Ok(Self::Closure(Closure {
                function_name: name,
                bound_function: Some(function),
                captures: Vec::new(),
            })),
            Value::Closure(closure) => Ok(Self::bind_closure(closure)),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Err(arrayfun_flow(
                "arrayfun: expected function handle or builtin name, not a scalar value",
            )),
            other => Err(arrayfun_flow(format!(
                "arrayfun: expected function handle or builtin name, got {other:?}"
            ))),
        }
    }

    pub(in crate::builtins::acceleration::gpu::arrayfun) fn from_text(
        text: &str,
    ) -> BuiltinResult<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(arrayfun_flow(
                "arrayfun: expected function handle or builtin name, got empty string",
            ));
        }
        let name = match trimmed.strip_prefix('@') {
            Some(name) if name.trim().is_empty() => {
                return Err(arrayfun_flow("arrayfun: empty function handle"));
            }
            Some(name) => name.trim().to_string(),
            None => trimmed.to_ascii_lowercase(),
        };
        Self::resolve_name(name)
    }

    fn resolve_name(name: String) -> BuiltinResult<Self> {
        if let Some(callable) = Self::resolved_semantic_handle(&name) {
            return Ok(callable);
        }
        if crate::is_well_formed_qualified_name(&name) {
            return Ok(Self::ExternalName { name });
        }
        Ok(Self::resolved_builtin(&name).unwrap_or(Self::DynamicName { name }))
    }

    fn from_external_name(name: String) -> BuiltinResult<Self> {
        if let Some(callable) = Self::resolved_semantic_handle(&name) {
            return Ok(callable);
        }
        if crate::is_well_formed_qualified_name(&name) {
            return Ok(Self::ExternalName { name });
        }
        Ok(Self::resolved_builtin(&name).unwrap_or(Self::DynamicName { name }))
    }

    fn bind_closure(mut closure: Closure) -> Self {
        if closure.bound_function.is_none() {
            closure.bound_function =
                user_functions::resolve_semantic_function_by_name(&closure.function_name);
        }
        Self::Closure(closure)
    }
}
