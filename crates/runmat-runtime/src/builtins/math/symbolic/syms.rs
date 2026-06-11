//! MATLAB-compatible scalar `syms` builtin for declaring symbolic variables.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    SymbolicExpr, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, workspace, BuiltinResult, RuntimeError};

use super::{empty_return_value, is_valid_identifier, text_scalar};

const BUILTIN_NAME: &str = "syms";

const SYMS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ans",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty return value for command declarations.",
}];

const SYMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Symbol names and optional assumption keywords.",
}];

const SYMS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "syms name ...",
    inputs: &SYMS_INPUTS,
    outputs: &SYMS_OUTPUT,
}];

const SYMS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.SYMS.EMPTY",
        identifier: Some("RunMat:syms:EmptyDeclaration"),
        when: "No symbol names were supplied.",
        message: "syms: expected at least one symbolic variable name",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYMS.INVALID_NAME",
        identifier: Some("RunMat:syms:InvalidName"),
        when: "A declaration token is not a valid MATLAB identifier or assumption keyword.",
        message: "syms: invalid symbolic variable name",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYMS.WORKSPACE",
        identifier: Some("RunMat:syms:WorkspaceUnavailable"),
        when: "The active runtime cannot assign to the caller workspace.",
        message: "syms: workspace assignment unavailable",
    },
];

pub const SYMS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYMS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SYMS_ERRORS,
};

#[runtime_builtin(
    name = "syms",
    category = "math/symbolic",
    summary = "Declare scalar symbolic variables in the active workspace.",
    keywords = "syms,symbolic,variable,workspace",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::math::symbolic::syms::SYMS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::symbolic::syms"
)]
async fn syms_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let names = declared_symbol_names(&args)?;
    if names.is_empty() {
        return Err(syms_error(&SYMS_ERRORS[0]));
    }
    for name in names {
        workspace::assign(&name, Value::Symbolic(SymbolicExpr::variable(&name))).map_err(
            |err| {
                syms_error_with_message(
                    &SYMS_ERRORS[2],
                    format!("{}: {}", SYMS_ERRORS[2].message, err),
                )
            },
        )?;
    }
    Ok(empty_return_value())
}

fn declared_symbol_names(args: &[Value]) -> BuiltinResult<Vec<String>> {
    let mut names = Vec::new();
    for arg in args {
        let Some(text) = text_scalar(arg) else {
            return Err(syms_error(&SYMS_ERRORS[1]));
        };
        for token in text.split_whitespace() {
            if names.is_empty() || !is_assumption_keyword(token) {
                if !is_valid_identifier(token) {
                    return Err(syms_error_with_message(
                        &SYMS_ERRORS[1],
                        format!("{}: '{token}'", SYMS_ERRORS[1].message),
                    ));
                }
                names.push(token.to_string());
            }
        }
    }
    Ok(names)
}

fn is_assumption_keyword(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "real"
            | "positive"
            | "negative"
            | "integer"
            | "rational"
            | "clear"
            | "finite"
            | "nonzero"
            | "nonnegative"
            | "nonpositive"
            | "complex"
    )
}

fn syms_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    syms_error_with_message(error, error.message)
}

fn syms_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(message.to_string()).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_and_skips_assumptions_after_first_symbol() {
        let names = declared_symbol_names(&[
            Value::from("x"),
            Value::from("real"),
            Value::from("h"),
            Value::from("positive"),
        ])
        .expect("names");

        assert_eq!(names, vec!["x".to_string(), "h".to_string()]);
    }

    #[test]
    fn rejects_identifier_with_leading_underscore() {
        let err = declared_symbol_names(&[Value::from("_x")])
            .expect_err("invalid identifier should error");

        assert_eq!(err.identifier.as_deref(), Some("RunMat:syms:InvalidName"));
    }
}
