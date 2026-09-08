use crate::builtins::common::shape::value_numel;
use crate::BuiltinResult;
use runmat_value::Value;

use super::super::error;

#[derive(Clone, Copy)]
pub(in crate::builtins::cells::core::cellfun) enum CellfunShorthand {
    IsClass,
    ProdOfSize,
}

impl CellfunShorthand {
    pub(super) fn parse_bare_value(value: &Value) -> Option<Self> {
        let text = match value {
            Value::String(text) => text.as_str(),
            Value::CharArray(value) if value.rows == 1 => {
                return Self::parse_name(&value.data.iter().collect::<String>());
            }
            Value::StringArray(value) if value.data.len() == 1 => value.data[0].as_str(),
            _ => return None,
        };
        Self::parse_name(text)
    }

    fn parse_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "isclass" => Some(Self::IsClass),
            "prodofsize" => Some(Self::ProdOfSize),
            _ => None,
        }
    }

    pub(super) async fn call(self, arguments: &[Value]) -> BuiltinResult<Value> {
        match self {
            Self::ProdOfSize => prod_of_size(arguments).await,
            Self::IsClass => is_class(arguments).await,
        }
    }
}

async fn prod_of_size(arguments: &[Value]) -> BuiltinResult<Value> {
    let value = arguments
        .first()
        .ok_or_else(|| error::invalid("cellfun: prodofsize requires one input"))?;
    Ok(Value::Num(value_numel(value).await? as f64))
}

async fn is_class(arguments: &[Value]) -> BuiltinResult<Value> {
    let [value, requested, ..] = arguments else {
        return Err(error::invalid(
            "cellfun: 'isclass' requires a class name argument",
        ));
    };
    let requested = class_identity(requested)
        .ok_or_else(|| error::invalid("cellfun: class name must be a string scalar"))?;
    let actual = crate::builtins::introspection::class::class_identity_for_value(value);
    Ok(Value::Bool(actual == requested))
}

fn class_identity(value: &Value) -> Option<runmat_types::ClassIdentity> {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::CharArray(value) if value.rows == 1 => value.data.iter().collect(),
        Value::StringArray(value) if value.data.len() == 1 => value.data[0].clone(),
        _ => return None,
    };
    runmat_types::ClassIdentity::new(text.trim()).ok()
}
