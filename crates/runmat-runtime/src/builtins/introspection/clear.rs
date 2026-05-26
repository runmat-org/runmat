//! MATLAB-compatible `clear` builtin for workspace variables.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, workspace, BuiltinResult};

const CLEAR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ans",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty 0x0 return value.",
}];

const CLEAR_SIG_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const CLEAR_SIG_NAME_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variable names to clear. 'all' clears entire workspace.",
}];

const CLEAR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "clear()",
        inputs: &CLEAR_SIG_NO_INPUTS,
        outputs: &CLEAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "clear(name, ...)",
        inputs: &CLEAR_SIG_NAME_INPUTS,
        outputs: &CLEAR_OUTPUT,
    },
];

const CLEAR_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.CLEAR.CHAR_ROWS",
        identifier: None,
        when: "Character-array argument is not a row vector.",
        message: "clear: character array inputs must be a row vector or scalar text value",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEAR.NAME_ARG_TYPE",
        identifier: None,
        when: "A target argument is not a string scalar/array or row character vector.",
        message: "clear: expected variable names as character vectors or string scalars",
    },
];

pub const CLEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLEAR_ERRORS,
};

#[runtime_builtin(
    name = "clear",
    category = "introspection",
    summary = "Clear variables from the active workspace.",
    keywords = "clear,workspace,variables",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::clear::CLEAR_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::clear"
)]
async fn clear_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        workspace::clear().map_err(clear_error)?;
        return Ok(empty_return_value());
    }

    let mut names = Vec::new();
    for arg in &args {
        collect_clear_targets(arg, &mut names)?;
    }

    if names.is_empty() || names.iter().any(|name| name.eq_ignore_ascii_case("all")) {
        workspace::clear().map_err(clear_error)?;
    } else {
        for name in names {
            workspace::remove(&name).map_err(clear_error)?;
        }
    }

    Ok(empty_return_value())
}

fn clear_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("clear").build()
}

fn collect_clear_targets(arg: &Value, out: &mut Vec<String>) -> BuiltinResult<()> {
    match arg {
        Value::String(text) => {
            out.push(text.trim().to_string());
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows > 1 {
                return Err(clear_error(
                    "clear: character array inputs must be a row vector or scalar text value",
                ));
            }
            out.push(chars.data.iter().collect::<String>().trim().to_string());
            Ok(())
        }
        Value::StringArray(array) => {
            for text in &array.data {
                out.push(text.trim().to_string());
            }
            Ok(())
        }
        _ => Err(clear_error(
            "clear: expected variable names as character vectors or string scalars",
        )),
    }
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use once_cell::sync::Lazy;
    use runmat_builtins::Value;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Mutex;

    static CLEAR_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn test_guard() -> (
        std::sync::MutexGuard<'static, ()>,
        std::sync::MutexGuard<'static, ()>,
    ) {
        let workspace = crate::workspace::test_guard();
        let clear = CLEAR_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        (workspace, clear)
    }

    fn clear_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::clear_builtin(args))
    }

    fn ensure_test_resolver() {
        crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
            lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
            snapshot: || {
                TEST_WORKSPACE.with(|slot| {
                    let mut entries: Vec<(String, Value)> =
                        slot.borrow().clone().into_iter().collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    entries
                })
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
    }

    fn set_workspace(entries: &[(&str, Value)]) {
        TEST_WORKSPACE.with(|slot| {
            let mut map = slot.borrow_mut();
            map.clear();
            for (name, value) in entries {
                map.insert((*name).to_string(), value.clone());
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clear_without_args_clears_workspace() {
        let (_workspace_guard, _clear_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[("x", Value::Num(1.0)), ("y", Value::Num(2.0))]);
        clear_builtin(vec![]).expect("clear");
        let snapshot = crate::workspace::snapshot().unwrap_or_default();
        assert!(snapshot.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clear_named_target_removes_only_named_variable() {
        let (_workspace_guard, _clear_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[("x", Value::Num(1.0)), ("y", Value::Num(2.0))]);
        clear_builtin(vec![Value::from("x")]).expect("clear");
        let snapshot = crate::workspace::snapshot().unwrap_or_default();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].0, "y");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clear_all_keyword_clears_workspace() {
        let (_workspace_guard, _clear_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[("x", Value::Num(1.0)), ("y", Value::Num(2.0))]);
        clear_builtin(vec![Value::from("all")]).expect("clear");
        let snapshot = crate::workspace::snapshot().unwrap_or_default();
        assert!(snapshot.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clear_rejects_non_string_target() {
        let (_workspace_guard, _clear_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[("x", Value::Num(1.0))]);
        let err = clear_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert!(err
            .to_string()
            .contains("expected variable names as character vectors or string scalars"));
    }
}
