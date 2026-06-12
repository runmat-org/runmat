//! MATLAB-compatible scalar `syms` builtin for declaring symbolic variables.

use runmat_builtins::{
    symbolic::{parse_symbolic_declaration, symbolic_declaration_tokens, SymbolicDeclaration},
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    SymbolicExpr, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, workspace, BuiltinResult, RuntimeError};

use super::{empty_return_value, text_scalar};

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
    let declarations = declared_symbols(&args)?;
    if declarations.is_empty() {
        return Err(syms_error(&SYMS_ERRORS[0]));
    }
    for declaration in declarations {
        for parameter in &declaration.parameters {
            assign_symbolic_value(parameter, SymbolicExpr::variable(parameter))?;
        }

        let value = if declaration.parameters.is_empty() {
            SymbolicExpr::variable(&declaration.name)
        } else {
            SymbolicExpr::function_reference(&declaration.name, declaration.parameters.clone())
        };
        assign_symbolic_value(&declaration.name, value)?;
    }
    Ok(empty_return_value())
}

fn declared_symbols(args: &[Value]) -> BuiltinResult<Vec<SymbolicDeclaration>> {
    let mut declarations = Vec::new();
    for arg in args {
        let Some(text) = text_scalar(arg) else {
            return Err(syms_error(&SYMS_ERRORS[1]));
        };
        for token in symbolic_declaration_tokens(&text) {
            if declarations.is_empty() || !is_assumption_keyword(token) {
                let declaration = parse_symbolic_declaration(token).map_err(|err| {
                    syms_error_with_message(
                        &SYMS_ERRORS[1],
                        format!("{}: '{token}' ({err})", SYMS_ERRORS[1].message),
                    )
                })?;
                declarations.push(declaration);
            }
        }
    }
    Ok(declarations)
}

fn assign_symbolic_value(name: &str, expr: SymbolicExpr) -> BuiltinResult<()> {
    workspace::assign(name, Value::Symbolic(expr)).map_err(|err| {
        syms_error_with_message(
            &SYMS_ERRORS[2],
            format!("{}: {}", SYMS_ERRORS[2].message, err),
        )
    })
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
    use futures::executor::block_on;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn test_workspace_guard() -> std::sync::MutexGuard<'static, ()> {
        let guard = crate::workspace::test_guard();
        crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
            lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
            snapshot: || {
                let mut entries: Vec<(String, Value)> =
                    TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries
            },
            globals: || Vec::new(),
            assign: Some(|name, value| {
                TEST_WORKSPACE.with(|slot| {
                    slot.borrow_mut().insert(name.to_string(), value);
                });
                Ok(())
            }),
            clear: Some(|| {
                TEST_WORKSPACE.with(|slot| slot.borrow_mut().clear());
                Ok(())
            }),
            remove: Some(|name| {
                TEST_WORKSPACE.with(|slot| {
                    slot.borrow_mut().remove(name);
                });
                Ok(())
            }),
        });
        TEST_WORKSPACE.with(|slot| slot.borrow_mut().clear());
        guard
    }

    #[test]
    fn parses_names_and_skips_assumptions_after_first_symbol() {
        let declarations = declared_symbols(&[
            Value::from("x"),
            Value::from("real"),
            Value::from("h"),
            Value::from("positive"),
        ])
        .expect("names");

        assert_eq!(
            declarations
                .into_iter()
                .map(|declaration| declaration.name)
                .collect::<Vec<_>>(),
            vec!["x".to_string(), "h".to_string()]
        );
    }

    #[test]
    fn parses_function_declarations_and_parameters() {
        let declarations = declared_symbols(&[Value::from("Y(X) f(x, y)")]).expect("declarations");

        assert_eq!(declarations[0].name, "Y");
        assert_eq!(declarations[0].parameters, vec!["X"]);
        assert_eq!(declarations[1].name, "f");
        assert_eq!(declarations[1].parameters, vec!["x", "y"]);
    }

    #[test]
    fn assigns_symbolic_function_declaration_to_workspace() {
        let _guard = test_workspace_guard();

        block_on(syms_builtin(vec![Value::from("Y(X)")])).expect("syms");

        assert_eq!(crate::workspace::lookup("X").unwrap().to_string(), "X");
        assert_eq!(crate::workspace::lookup("Y").unwrap().to_string(), "Y(X)");
    }

    #[test]
    fn rejects_identifier_with_leading_underscore() {
        let err =
            declared_symbols(&[Value::from("_x")]).expect_err("invalid identifier should error");

        assert_eq!(err.identifier.as_deref(), Some("RunMat:syms:InvalidName"));
    }
}
