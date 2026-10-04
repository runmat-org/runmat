//! Durability eligibility for the existing exact execution value payload.
//!
//! A transportable value is not necessarily a persistent one: callable names,
//! execution handles, and registered classes can depend on a session or host.

use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_value::Value;

use super::{
    decode::decode_durable_inline_value, encode::encode_durable_inline_value, ValueCodecError,
};

pub fn encode_durable_value(value: &Value) -> Result<ValuePayload, ValueCodecError> {
    let payload = encode_durable_inline_value(value)?;
    require_durable(&payload, "$")?;
    payload
        .validate_for_durable_storage(runmat_execution::value::ValueLimits::default())
        .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
    Ok(payload)
}

pub fn decode_durable_value(payload: &ValuePayload) -> Result<Value, ValueCodecError> {
    require_durable(payload, "$")?;
    decode_durable_inline_value(payload)
}

fn require_durable(payload: &ValuePayload, path: &str) -> Result<(), ValueCodecError> {
    let ValuePayload::Inline(value) = payload else {
        return Err(ValueCodecError::unsupported(
            path,
            "execution object and distributed handles are not durable preferences",
        ));
    };
    match value.as_ref() {
        InlineValue::Callable(_) => Err(ValueCodecError::unsupported(
            path,
            "callable identities and captures are bound to an execution context",
        )),
        InlineValue::ImmutableValueClass(_) | InlineValue::Symbolic(_) => Err(
            ValueCodecError::unsupported(path, "a registered durable class codec is required"),
        ),
        InlineValue::Null => Err(ValueCodecError::unsupported(
            path,
            "null is not a RunMat preference value",
        )),
        InlineValue::Cell { values, .. } => {
            for (index, value) in values.iter().enumerate() {
                require_durable(value, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        InlineValue::Struct(fields) => {
            for field in fields {
                require_durable(&field.value, &format!("{path}.{}", field.name))?;
            }
            Ok(())
        }
        InlineValue::DurablePlainObject(value) => {
            for field in &value.properties {
                require_durable(&field.value, &format!("{path}.{}", field.name))?;
            }
            Ok(())
        }
        InlineValue::DurableStructArray(value) => {
            for field in &value.fields {
                for (index, item) in field.values.iter().enumerate() {
                    require_durable(item, &format!("{path}.{}[{index}]", field.name))?;
                }
            }
            Ok(())
        }
        InlineValue::DurableObjectArray(value) => {
            for (index, object) in value.elements.iter().enumerate() {
                for field in &object.properties {
                    require_durable(&field.value, &format!("{path}[{index}].{}", field.name))?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
