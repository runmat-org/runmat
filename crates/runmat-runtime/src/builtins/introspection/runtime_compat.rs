//! MATLAB runtime/version compatibility helpers.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const RUNMAT_VERSION: &str = env!("CARGO_PKG_VERSION");
const MATLAB_COMPAT_VERSION: &str = "9.15";
const MATLAB_COMPAT_RELEASE: &str = "R2023b";

const VALUE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Version or logical result.",
}];

const ANY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Version query arguments.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = runtimeCompat(args...)",
    inputs: &ANY_INPUTS,
    outputs: &VALUE_OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTIME_COMPAT.INVALID_INPUT",
    identifier: Some("RunMat:runtimeCompat:InvalidInput"),
    when: "Arguments are malformed or unsupported.",
    message: "runtime compatibility helper: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const RUNTIME_COMPAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ISDEPLOYED_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "isdeployed accepts no arguments and has no integer data, control, output-class, or provider surface." };

#[runtime_builtin(
    name = "version",
    category = "introspection",
    summary = "Return RunMat's MATLAB-compatible version string.",
    keywords = "version,release,environment,compatibility",
    descriptor(crate::builtins::introspection::runtime_compat::RUNTIME_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::runtime_compat"
)]
fn version_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Ok(Value::from(RUNMAT_VERSION));
    }
    if args.len() != 1 {
        return Err(error("version: expected zero or one argument"));
    }
    match text_scalar(&args[0])?.to_ascii_lowercase().as_str() {
        "-release" => Ok(Value::from(MATLAB_COMPAT_RELEASE)),
        "-description" => Ok(Value::from(format!(
            "RunMat {RUNMAT_VERSION} (MATLAB compatibility {MATLAB_COMPAT_RELEASE})"
        ))),
        "-java" => Ok(Value::from("")),
        other => Err(error(format!("version: unsupported option '{other}'"))),
    }
}

#[runtime_builtin(
    name = "verLessThan",
    category = "introspection",
    summary = "Compare the MATLAB compatibility version against a required version.",
    keywords = "verLessThan,version,compatibility",
    descriptor(crate::builtins::introspection::runtime_compat::RUNTIME_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::runtime_compat"
)]
fn ver_less_than_builtin(product: Value, version: Value) -> BuiltinResult<Value> {
    let product = text_scalar(&product)?.to_ascii_lowercase();
    let requested = text_scalar(&version)?;
    if product != "matlab" {
        return Ok(Value::Bool(false));
    }
    Ok(Value::Bool(
        compare_versions(MATLAB_COMPAT_VERSION, &requested) < 0,
    ))
}

#[runtime_builtin(
    name = "isdeployed",
    category = "introspection",
    summary = "Return false because RunMat is not MATLAB Compiler deployed code.",
    keywords = "isdeployed,deployment,runtime",
    descriptor(crate::builtins::introspection::runtime_compat::RUNTIME_COMPAT_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::runtime_compat::ISDEPLOYED_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::runtime_compat"
)]
fn isdeployed_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(error("isdeployed: expected no input arguments"));
    }
    Ok(Value::Bool(false))
}

fn text_scalar(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(error("runtime compatibility helper: expected text scalar")),
    }
}

fn compare_versions(left: &str, right: &str) -> i8 {
    let left_parts = version_parts(left);
    let right_parts = version_parts(right);
    let len = left_parts.len().max(right_parts.len());
    for idx in 0..len {
        let l = *left_parts.get(idx).unwrap_or(&0);
        let r = *right_parts.get(idx).unwrap_or(&0);
        if l < r {
            return -1;
        }
        if l > r {
            return 1;
        }
    }
    0
}

fn version_parts(value: &str) -> Vec<u64> {
    value
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect()
}

fn error(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("runtime compatibility helper");
    if let Some(identifier) = ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_options_return_text() {
        assert!(matches!(
            version_builtin(Vec::new()).unwrap(),
            Value::String(_)
        ));
        assert_eq!(
            version_builtin(vec![Value::from("-release")]).unwrap(),
            Value::from(MATLAB_COMPAT_RELEASE)
        );
    }

    #[test]
    fn deployed_and_version_comparison_are_stable() {
        assert_eq!(isdeployed_builtin(Vec::new()).unwrap(), Value::Bool(false));
        assert_eq!(
            ver_less_than_builtin(Value::from("matlab"), Value::from("99.0")).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            ver_less_than_builtin(Value::from("simulink"), Value::from("99.0")).unwrap(),
            Value::Bool(false)
        );
    }
}
