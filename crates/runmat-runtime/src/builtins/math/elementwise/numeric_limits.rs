//! MATLAB-compatible numeric limit query builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const INPUTS_CLASS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "classname",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Numeric class name.",
}];

const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Limit value.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "value = limit()",
        inputs: &[],
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "value = limit(classname)",
        inputs: &INPUTS_CLASS,
        outputs: &OUTPUT_VALUE,
    },
];

const ERROR_INVALID_CLASS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUMERIC_LIMITS.INVALID_CLASS",
    identifier: Some("RunMat:numericLimits:InvalidClass"),
    when: "The requested class is not supported by the limit query.",
    message: "numeric limit: unsupported class",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_CLASS];

pub const NUMERIC_LIMIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "intmax",
    category = "math/elementwise",
    summary = "Return the largest value of an integer class.",
    keywords = "intmax,integer,limits",
    descriptor(crate::builtins::math::elementwise::numeric_limits::NUMERIC_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn intmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "int32")?;
    match class.as_str() {
        "int8" => Ok(Value::Int(IntValue::I8(i8::MAX))),
        "int16" => Ok(Value::Int(IntValue::I16(i16::MAX))),
        "int32" => Ok(Value::Int(IntValue::I32(i32::MAX))),
        "int64" => Ok(Value::Int(IntValue::I64(i64::MAX))),
        "uint8" => Ok(Value::Int(IntValue::U8(u8::MAX))),
        "uint16" => Ok(Value::Int(IntValue::U16(u16::MAX))),
        "uint32" => Ok(Value::Int(IntValue::U32(u32::MAX))),
        "uint64" => Ok(Value::Int(IntValue::U64(u64::MAX))),
        _ => Err(limit_error(
            "intmax",
            format!("unsupported integer class '{class}'"),
        )),
    }
}

#[runtime_builtin(
    name = "intmin",
    category = "math/elementwise",
    summary = "Return the smallest value of an integer class.",
    keywords = "intmin,integer,limits",
    descriptor(crate::builtins::math::elementwise::numeric_limits::NUMERIC_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn intmin_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "int32")?;
    match class.as_str() {
        "int8" => Ok(Value::Int(IntValue::I8(i8::MIN))),
        "int16" => Ok(Value::Int(IntValue::I16(i16::MIN))),
        "int32" => Ok(Value::Int(IntValue::I32(i32::MIN))),
        "int64" => Ok(Value::Int(IntValue::I64(i64::MIN))),
        "uint8" => Ok(Value::Int(IntValue::U8(0))),
        "uint16" => Ok(Value::Int(IntValue::U16(0))),
        "uint32" => Ok(Value::Int(IntValue::U32(0))),
        "uint64" => Ok(Value::Int(IntValue::U64(0))),
        _ => Err(limit_error(
            "intmin",
            format!("unsupported integer class '{class}'"),
        )),
    }
}

#[runtime_builtin(
    name = "realmax",
    category = "math/elementwise",
    summary = "Return the largest finite floating-point value.",
    keywords = "realmax,float,limits,double,single",
    descriptor(crate::builtins::math::elementwise::numeric_limits::NUMERIC_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn realmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "double")?;
    match class.as_str() {
        "double" => Ok(Value::Num(f64::MAX)),
        "single" => Ok(Value::Num(f32::MAX as f64)),
        _ => Err(limit_error(
            "realmax",
            format!("unsupported float class '{class}'"),
        )),
    }
}

#[runtime_builtin(
    name = "realmin",
    category = "math/elementwise",
    summary = "Return the smallest positive normalized floating-point value.",
    keywords = "realmin,float,limits,double,single",
    descriptor(crate::builtins::math::elementwise::numeric_limits::NUMERIC_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn realmin_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "double")?;
    match class.as_str() {
        "double" => Ok(Value::Num(f64::MIN_POSITIVE)),
        "single" => Ok(Value::Num(f32::MIN_POSITIVE as f64)),
        _ => Err(limit_error(
            "realmin",
            format!("unsupported float class '{class}'"),
        )),
    }
}

#[runtime_builtin(
    name = "flintmax",
    category = "math/elementwise",
    summary = "Return the largest consecutive integer in a floating-point class.",
    keywords = "flintmax,float,limits,double,single",
    descriptor(crate::builtins::math::elementwise::numeric_limits::NUMERIC_LIMIT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn flintmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let class = parse_class(rest.first(), "double")?;
    match class.as_str() {
        "double" => Ok(Value::Num(2f64.powi(53))),
        "single" => Ok(Value::Num(2f64.powi(24))),
        _ => Err(limit_error(
            "flintmax",
            format!("unsupported float class '{class}'"),
        )),
    }
}

fn parse_class(value: Option<&Value>, default: &str) -> BuiltinResult<String> {
    match value {
        None => Ok(default.to_string()),
        Some(Value::String(text)) => Ok(normalize_class(text)),
        Some(Value::CharArray(chars)) if chars.rows == 1 => {
            Ok(normalize_class(&chars.data.iter().collect::<String>()))
        }
        Some(Value::StringArray(array)) if array.data.len() == 1 => {
            Ok(normalize_class(&array.data[0]))
        }
        Some(_) => Err(limit_error(
            "numeric limit",
            "class name must be a string scalar or character vector",
        )),
    }
}

fn normalize_class(text: &str) -> String {
    match text.trim().to_ascii_lowercase().as_str() {
        "float" => "single".to_string(),
        "int" => "int32".to_string(),
        other => other.to_string(),
    }
}

fn limit_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = ERROR_INVALID_CLASS.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_limits_support_common_classes() {
        assert_eq!(
            intmax_builtin(Vec::new()).unwrap(),
            Value::Int(IntValue::I32(i32::MAX))
        );
        assert_eq!(
            intmin_builtin(vec![Value::from("uint16")]).unwrap(),
            Value::Int(IntValue::U16(0))
        );
        assert_eq!(
            intmax_builtin(vec![Value::from("uint32")]).unwrap(),
            Value::Int(IntValue::U32(u32::MAX))
        );
    }

    #[test]
    fn floating_limits_support_single_and_double() {
        assert_eq!(realmax_builtin(Vec::new()).unwrap(), Value::Num(f64::MAX));
        assert_eq!(
            realmin_builtin(vec![Value::from("single")]).unwrap(),
            Value::Num(f32::MIN_POSITIVE as f64)
        );
        assert_eq!(
            flintmax_builtin(vec![Value::from("single")]).unwrap(),
            Value::Num(2f64.powi(24))
        );
    }
}
