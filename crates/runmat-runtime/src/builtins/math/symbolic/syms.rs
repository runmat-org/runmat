//! MATLAB-compatible scalar `syms` builtin for declaring symbolic variables.
//! Runtime-owned effect boundary: accepted declarations contain only text, so
//! there are no resident data operands to gather or invalidate. Invalid resident
//! inputs are rejected by declaration admission before provider/workspace access.

use runmat_builtins::{
    BuiltinErrorDescriptor, SYMS_DESCRIPTOR, SYMS_ERRORS, SYMS_ERROR_TOO_MANY_OUTPUTS,
};

use runmat_builtins::{
    parse_symbolic_declaration, symbolic_declaration_tokens, SymbolicDeclaration,
};
use runmat_value::{SymbolicExpr, Value};

use crate::{build_runtime_error, workspace, BuiltinResult, RuntimeError};

use super::{empty_return_value, text_scalar};

const BUILTIN_NAME: &str = "syms";

#[runmat_macros::runtime_builtin(
    name = "syms",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::symbolic::syms"
)]
async fn syms_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &SYMS_DESCRIPTOR,
        &SYMS_ERROR_TOO_MANY_OUTPUTS,
        "syms",
    )?;
    let declarations = declared_symbols(args.view())?;
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

fn declared_symbols(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<SymbolicDeclaration>> {
    let mut declarations = Vec::new();
    for arg in args {
        if matches!(arg, Value::CharArray(chars) if chars.shape.iter().skip(2).any(|dimension| *dimension != 1))
        {
            return Err(syms_error(&SYMS_ERRORS[1]));
        }
        let Some(text) = text_scalar(arg).map_err(|source| {
            let mut error = syms_error(&SYMS_ERRORS[1]);
            error.source = Some(Box::new(source));
            error
        })?
        else {
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
    workspace::try_assign(name, Value::Symbolic(expr)).map_err(syms_workspace_flow)
}

fn syms_workspace_flow(mut error: RuntimeError) -> RuntimeError {
    error.context = error.context.with_builtin(BUILTIN_NAME);
    error
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
pub(crate) async fn syms_builtin_source(args: Vec<Value>) -> BuiltinResult<Value> {
    syms_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?).await
}

#[cfg(test)]
mod tests {
    fn declared_symbols(args: &[Value]) -> BuiltinResult<Vec<SymbolicDeclaration>> {
        super::declared_symbols(crate::call::FunctionArgumentView::from_slice(args))
    }
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
    fn validation_and_output_arity_fail_before_any_workspace_write() {
        let _workspace = test_workspace_guard();
        let error = block_on(syms_builtin_source(vec![
            Value::from("x"),
            Value::from("_bad"),
        ]))
        .unwrap_err();
        assert_eq!(error.identifier(), SYMS_ERRORS[1].identifier);
        assert!(crate::workspace::lookup("x").is_none());
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = block_on(syms_builtin_source(vec![Value::from("x")])).unwrap_err();
        assert_eq!(error.identifier(), SYMS_ERROR_TOO_MANY_OUTPUTS.identifier);
        assert!(crate::workspace::lookup("x").is_none());
    }

    #[test]
    fn resident_declaration_rejects_without_provider_or_workspace_effects() {
        let _workspace = test_workspace_guard();
        runmat_accelerate::ensure_residency_hooks();
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let handle = allocation.handle;
        assert!(crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle).is_none());
        runmat_accelerate_api::mark_residency(&handle);
        let error = block_on(syms_builtin_source(vec![
            Value::from("x"),
            Value::GpuTensor(handle.clone()),
        ]))
        .unwrap_err();
        assert_eq!(error.identifier(), SYMS_ERRORS[1].identifier);
        assert!(workspace::lookup("x").is_none());
        assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
        assert_eq!(allocation.probe.retirements(), 0);
        handle.release().unwrap();
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn zero_outputs_still_declares_and_placeholder_is_empty_double() {
        let _workspace = test_workspace_guard();
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let value = block_on(syms_builtin_source(vec![Value::from("x h")])).unwrap();
        assert_eq!(crate::workspace::lookup("x").unwrap().to_string(), "x");
        assert_eq!(crate::workspace::lookup("h").unwrap().to_string(), "h");
        let Value::Tensor(tensor) = value else {
            panic!("empty double placeholder")
        };
        assert_eq!(tensor.shape(), &[0, 0]);
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

        block_on(syms_builtin_source(vec![Value::from("Y(X)")])).expect("syms");

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
