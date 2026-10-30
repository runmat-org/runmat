//! MATLAB runtime/version compatibility helpers.

use runmat_builtins::{
    BuiltinErrorDescriptor, ISDEPLOYED_INVALID_INPUT, VERSION_INVALID_INPUT,
    VER_LESS_THAN_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const RUNMAT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy)]
enum CompatibilityOperation {
    Version,
    VerLessThan,
    IsDeployed,
}

impl CompatibilityOperation {
    fn descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Version => &VERSION_INVALID_INPUT,
            Self::VerLessThan => &VER_LESS_THAN_INVALID_INPUT,
            Self::IsDeployed => &ISDEPLOYED_INVALID_INPUT,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Version => "version",
            Self::VerLessThan => "verLessThan",
            Self::IsDeployed => "isdeployed",
        }
    }
}

#[runtime_builtin(
    name = "version",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::runtime_compat"
)]
fn version_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Ok(Value::from(RUNMAT_VERSION));
    }
    if args.len() != 1 {
        return Err(error(
            CompatibilityOperation::Version,
            "version: expected zero or one argument",
        ));
    }
    match text_scalar(&args[0], CompatibilityOperation::Version)?
        .to_ascii_lowercase()
        .as_str()
    {
        "-release" => Ok(Value::from(
            crate::compatibility::MATLAB_COMPATIBILITY_RELEASE,
        )),
        "-description" => Ok(Value::from(format!(
            "RunMat {RUNMAT_VERSION} (MATLAB compatibility {})",
            crate::compatibility::MATLAB_COMPATIBILITY_RELEASE
        ))),
        "-java" => Ok(Value::from("")),
        other => Err(error(
            CompatibilityOperation::Version,
            format!("version: unsupported option '{other}'"),
        )),
    }
}

#[runtime_builtin(
    name = "verLessThan",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::runtime_compat"
)]
fn ver_less_than_builtin(product: Value, version: Value) -> BuiltinResult<Value> {
    let product = text_scalar(&product, CompatibilityOperation::VerLessThan)?.to_ascii_lowercase();
    let requested = text_scalar(&version, CompatibilityOperation::VerLessThan)?;
    if product != "matlab" {
        return Ok(Value::Bool(false));
    }
    Ok(Value::Bool(
        compare_versions(
            crate::compatibility::MATLAB_COMPATIBILITY_VERSION,
            &requested,
        ) < 0,
    ))
}

#[runtime_builtin(
    name = "isdeployed",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::runtime_compat"
)]
fn isdeployed_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(error(
            CompatibilityOperation::IsDeployed,
            "isdeployed: expected no input arguments",
        ));
    }
    Ok(Value::Bool(false))
}

fn text_scalar(value: &Value, operation: CompatibilityOperation) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(chars) if chars.rows == 1 => {
            super::text::decode_utf16_with_error(chars.data(), || {
                error(operation, "compatibility operand is not valid UTF-16 text")
            })
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        _ => Err(error(
            operation,
            "runtime compatibility helper: expected text scalar",
        )),
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

fn error(operation: CompatibilityOperation, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(operation.name());
    if let Some(identifier) = operation.descriptor().identifier {
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
            version_builtin(crate::builtins::common::test_support::builtin_arguments(
                Vec::new()
            ))
            .unwrap(),
            Value::String(_)
        ));
        assert_eq!(
            version_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::from("-release")]
            ))
            .unwrap(),
            Value::from(crate::compatibility::MATLAB_COMPATIBILITY_RELEASE)
        );
    }

    #[test]
    fn deployed_and_version_comparison_are_stable() {
        assert_eq!(
            isdeployed_builtin(crate::builtins::common::test_support::builtin_arguments(
                Vec::new()
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            ver_less_than_builtin(Value::from("matlab"), Value::from("99.0")).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            ver_less_than_builtin(
                Value::from("matlab"),
                Value::from(crate::compatibility::MATLAB_COMPATIBILITY_VERSION),
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            ver_less_than_builtin(Value::from("simulink"), Value::from("99.0")).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn runtime_version_controls_reject_integer_aliases() {
        assert!(
            version_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::Int(runmat_value::IntValue::I32(1))]
            ))
            .is_err()
        );
        assert!(ver_less_than_builtin(
            Value::Int(runmat_value::IntValue::I32(1)),
            Value::from(crate::compatibility::MATLAB_COMPATIBILITY_VERSION),
        )
        .is_err());
        assert!(ver_less_than_builtin(
            Value::from("matlab"),
            Value::Int(runmat_value::IntValue::I32(1)),
        )
        .is_err());
    }

    #[test]
    fn public_compatibility_forms_preserve_error_owner() {
        let version = version_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::Num(1.0)],
        ))
        .unwrap_err();
        let comparison = ver_less_than_builtin(Value::Bool(true), Value::from("1.0")).unwrap_err();
        let deployed = isdeployed_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
        )
        .unwrap_err();
        for error in [version, comparison, deployed] {
            assert_eq!(
                error.identifier(),
                Some("RunMat:runtimeCompat:InvalidInput")
            );
        }
    }
}
