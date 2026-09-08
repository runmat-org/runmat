use crate::user_functions;
use runmat_value::{Closure, Value};

use super::{CallableParseError, MappedCallable};

impl MappedCallable {
    pub(crate) fn parse(value: Value) -> Result<Self, CallableParseError> {
        match value {
            Value::String(text) => Self::parse_text(&text),
            Value::CharArray(value) if value.rows == 1 => {
                Self::parse_text(&value.data.iter().collect::<String>())
            }
            Value::CharArray(_) => Err(CallableParseError::CharacterNameMustBeRow),
            Value::StringArray(value) if value.data.len() == 1 => Self::parse_text(&value.data[0]),
            Value::StringArray(_) => Err(CallableParseError::StringNameMustBeScalar),
            Value::FunctionHandle(name) => Self::parse_text(&name),
            Value::ExternalFunctionHandle(name) => Ok(Self::resolve_name(name)),
            Value::BoundFunctionHandle { name, function } => Ok(Self::Closure(Closure {
                function_name: name,
                bound_function: Some(function),
                captures: Vec::new(),
            })),
            Value::Closure(closure) => Ok(Self::bind_closure(closure)),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Err(CallableParseError::ScalarValue),
            value => Err(CallableParseError::UnsupportedValue(value)),
        }
    }

    fn parse_text(text: &str) -> Result<Self, CallableParseError> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(CallableParseError::EmptyText);
        }
        let name = match trimmed.strip_prefix('@') {
            Some(name) if name.trim().is_empty() => return Err(CallableParseError::EmptyHandle),
            Some(name) => name.trim().to_string(),
            None => trimmed.to_ascii_lowercase(),
        };
        Ok(Self::resolve_name(name))
    }

    fn resolve_name(name: String) -> Self {
        if let Some(function) = user_functions::resolve_semantic_function_by_name(&name) {
            return Self::Closure(Closure {
                function_name: name,
                bound_function: Some(function),
                captures: Vec::new(),
            });
        }
        if crate::is_well_formed_qualified_name(&name) {
            return Self::ExternalName { name };
        }
        runmat_builtins::builtin_catalog_entry_by_name(&name)
            .map(|entry| Self::Builtin {
                identity: entry.identity,
            })
            .unwrap_or(Self::DynamicName { name })
    }

    fn bind_closure(mut closure: Closure) -> Self {
        if closure.bound_function.is_none() {
            closure.bound_function =
                user_functions::resolve_semantic_function_by_name(&closure.function_name);
        }
        Self::Closure(closure)
    }
}
