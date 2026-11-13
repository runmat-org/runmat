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
    pub(super) fn parse_bare_value(value: &Value) -> BuiltinResult<Option<Self>> {
        let text = match value {
            Value::String(runmat_value::StringElement::Text(text)) => text.as_str(),
            Value::CharArray(value) if value.rows == 1 => {
                return Ok(Self::parse_name(
                    &runmat_value::decode_utf16_code_units(value.data()).map_err(|source| {
                        let mut error = error::invalid(source.to_string());
                        error.source = Some(Box::new(source));
                        error
                    })?,
                ));
            }
            Value::StringArray(value)
                if value.data().len() == 1 && !value.data()[0].is_missing() =>
            {
                value.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
            }
            _ => return Ok(None),
        };
        Ok(Self::parse_name(text))
    }

    fn parse_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "isclass" => Some(Self::IsClass),
            "prodofsize" => Some(Self::ProdOfSize),
            _ => None,
        }
    }

    pub(super) async fn call(
        self,
        arguments: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Value> {
        match self {
            Self::ProdOfSize => prod_of_size(arguments).await,
            Self::IsClass => is_class(arguments).await,
        }
    }
}

async fn prod_of_size(arguments: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    let value = arguments
        .first()
        .ok_or_else(|| error::invalid("cellfun: prodofsize requires one input"))?;
    Ok(Value::Num(value_numel(value).await? as f64))
}

async fn is_class(arguments: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    if arguments.len() < 2 {
        return Err(error::invalid(
            "cellfun: 'isclass' requires a class name argument",
        ));
    };
    let value = &arguments[0];
    let requested = class_identity(&arguments[1])?
        .ok_or_else(|| error::invalid("cellfun: class name must be a string scalar"))?;
    let actual = crate::builtins::introspection::class_identity::class_identity_for_value(value);
    Ok(Value::Bool(actual == requested))
}

fn class_identity(value: &Value) -> BuiltinResult<Option<runmat_types::ClassIdentity>> {
    let text = match value {
        Value::String(runmat_value::StringElement::Text(text)) => text.to_string(),
        Value::CharArray(value) if value.rows == 1 => {
            runmat_value::decode_utf16_code_units(value.data()).map_err(|source| {
                let mut error = error::invalid(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?
        }
        Value::StringArray(value) if value.data().len() == 1 && !value.data()[0].is_missing() => {
            value.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .to_string()
        }
        _ => return Ok(None),
    };
    Ok(runmat_types::ClassIdentity::new(text.trim()).ok())
}
