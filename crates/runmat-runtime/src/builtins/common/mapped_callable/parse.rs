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
            Value::ExternalFunctionHandle(name) => {
                Ok(Self::resolve_name(name.into_name_unbounded()))
            }
            Value::BoundFunctionHandle(handle) => {
                let data = handle.into_data();
                Ok(Self::Closure(
                    crate::call::descriptor::RetainedClosureValue::new(Closure::new(
                        data.name,
                        Some(runmat_value::BoundFunctionBinding {
                            function: data.function,
                            authority: data.authority,
                        }),
                        data.source_role,
                        data.source_file,
                        Vec::new(),
                        None,
                    )),
                ))
            }
            Value::Closure(closure) => Ok(Self::Closure(
                crate::call::descriptor::RetainedClosureValue::new(Self::bind_closure(closure)),
            )),
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
            return Self::Closure(crate::call::descriptor::RetainedClosureValue::new(
                Closure::new(
                    name,
                    Some(user_functions::bound_function_binding_for(function)),
                    user_functions::source_function_for(function)
                        .map_or(runmat_types::FunctionSourceRole::Unknown, |info| info.role),
                    user_functions::source_file_for(function),
                    Vec::new(),
                    None,
                ),
            ));
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

    fn bind_closure(closure: Closure) -> Closure {
        if closure.binding.is_none() {
            if let Some(function) =
                user_functions::resolve_semantic_function_by_name(&closure.function_name)
            {
                return closure.bind_resolved(
                    user_functions::bound_function_binding_for(function),
                    user_functions::source_function_for(function)
                        .map_or(runmat_types::FunctionSourceRole::Unknown, |info| info.role),
                    user_functions::source_file_for(function),
                );
            }
        }
        closure
    }
}
