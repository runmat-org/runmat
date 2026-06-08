//! MATLAB-compatible `clearvars` builtin for workspace variables.

use std::collections::HashSet;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, workspace, BuiltinResult};

const CLEARVARS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ans",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty 0x0 return value.",
}];

const CLEARVARS_SIG_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const CLEARVARS_SIG_WORD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name_or_option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variable names and options such as '-except'.",
}];

const CLEARVARS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "clearvars()",
        inputs: &CLEARVARS_SIG_NO_INPUTS,
        outputs: &CLEARVARS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "clearvars(name_or_option, ...)",
        inputs: &CLEARVARS_SIG_WORD_INPUTS,
        outputs: &CLEARVARS_OUTPUT,
    },
];

const CLEARVARS_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.DUP_EXCEPT",
        identifier: None,
        when: "The '-except' option appears more than once.",
        message: "clearvars: duplicate -except option",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.UNSUPPORTED_OPTION",
        identifier: None,
        when: "An unknown option beginning with '-' is provided.",
        message: "clearvars: unsupported option",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.EXCEPT_REQUIRES_NAMES",
        identifier: None,
        when: "The '-except' option is provided without exclusion names.",
        message: "clearvars: -except requires at least one variable name",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.WORKSPACE_UNAVAILABLE",
        identifier: None,
        when: "Workspace snapshot is unavailable while processing '-except'.",
        message: "clearvars: workspace state unavailable",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.CHAR_ROWS",
        identifier: None,
        when: "Character-array argument is not a row vector.",
        message: "clearvars: character array inputs must be a row vector or scalar text value",
    },
    BuiltinErrorDescriptor {
        code: "RM.CLEARVARS.NAME_ARG_TYPE",
        identifier: None,
        when: "An argument is not a string scalar/array or row character vector.",
        message: "clearvars: expected variable names as character vectors or string scalars",
    },
];

pub const CLEARVARS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEARVARS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLEARVARS_ERRORS,
};

#[runtime_builtin(
    name = "clearvars",
    category = "introspection",
    summary = "Clear workspace variables with optional exclusions.",
    keywords = "clearvars,workspace,variables,except",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::introspection::type_resolvers::clearvars_type),
    descriptor(crate::builtins::introspection::clearvars::CLEARVARS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::clearvars"
)]
async fn clearvars_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        workspace::clear().map_err(clearvars_error)?;
        return Ok(empty_return_value());
    }

    let mut words = Vec::new();
    for arg in &args {
        collect_clearvars_words(arg, &mut words)?;
    }

    let mut targets = Vec::new();
    let mut exclusions = Vec::new();
    let mut saw_except = false;
    for word in words.iter() {
        let name = word.trim();
        if name.is_empty() {
            continue;
        }
        if name.eq_ignore_ascii_case("-except") {
            if saw_except {
                return Err(clearvars_error("clearvars: duplicate -except option"));
            }
            saw_except = true;
            continue;
        }
        if name.starts_with('-') {
            return Err(clearvars_error(format!(
                "clearvars: unsupported option '{name}'"
            )));
        }
        if saw_except {
            exclusions.push(name.to_string());
        } else {
            targets.push(name.to_string());
        }
    }

    if saw_except {
        if exclusions.is_empty() {
            return Err(clearvars_error(
                "clearvars: -except requires at least one variable name",
            ));
        }
        clear_except(&targets, &exclusions)?;
    } else {
        for name in targets {
            workspace::remove(&name).map_err(clearvars_error)?;
        }
    }

    Ok(empty_return_value())
}

fn clear_except(targets: &[String], exclusions: &[String]) -> BuiltinResult<()> {
    let keep: HashSet<&str> = exclusions.iter().map(String::as_str).collect();
    if targets.is_empty() {
        let snapshot = workspace::snapshot()
            .ok_or_else(|| clearvars_error("clearvars: workspace state unavailable"))?;
        for (name, _) in snapshot {
            if !keep.contains(name.as_str()) {
                workspace::remove(&name).map_err(clearvars_error)?;
            }
        }
    } else {
        for name in targets {
            if !keep.contains(name.as_str()) {
                workspace::remove(name).map_err(clearvars_error)?;
            }
        }
    }
    Ok(())
}

fn clearvars_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("clearvars")
        .build()
}

fn collect_clearvars_words(arg: &Value, out: &mut Vec<String>) -> BuiltinResult<()> {
    match arg {
        Value::String(text) => {
            out.push(text.trim().to_string());
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows > 1 {
                return Err(clearvars_error(
                    "clearvars: character array inputs must be a row vector or scalar text value",
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
        _ => Err(clearvars_error(
            "clearvars: expected variable names as character vectors or string scalars",
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

    static CLEARVARS_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn test_guard() -> (
        std::sync::MutexGuard<'static, ()>,
        std::sync::MutexGuard<'static, ()>,
    ) {
        let workspace = crate::workspace::test_guard();
        let clearvars = CLEARVARS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        (workspace, clearvars)
    }

    fn clearvars_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::clearvars_builtin(args))
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
    fn clearvars_named_targets_remove_only_selected_variables() {
        let (_workspace_guard, _clearvars_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[
            ("x", Value::Num(1.0)),
            ("y", Value::Num(2.0)),
            ("z", Value::Num(3.0)),
        ]);
        clearvars_builtin(vec![Value::from("x"), Value::from("z")]).expect("clearvars");
        let snapshot = crate::workspace::snapshot().unwrap_or_default();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].0, "y");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clearvars_except_keeps_excluded_variables() {
        let (_workspace_guard, _clearvars_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[
            ("x", Value::Num(1.0)),
            ("y", Value::Num(2.0)),
            ("z", Value::Num(3.0)),
        ]);
        clearvars_builtin(vec![
            Value::from("-except"),
            Value::from("y"),
            Value::from("z"),
        ])
        .expect("clearvars");
        let snapshot = crate::workspace::snapshot().unwrap_or_default();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0].0, "y");
        assert_eq!(snapshot[1].0, "z");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clearvars_rejects_duplicate_except_option() {
        let (_workspace_guard, _clearvars_guard) = test_guard();
        ensure_test_resolver();
        let err = clearvars_builtin(vec![
            Value::from("-except"),
            Value::from("-except"),
            Value::from("x"),
        ])
        .expect_err("expected duplicate -except error");
        assert!(err.to_string().contains("duplicate -except option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clearvars_rejects_non_string_arguments() {
        let (_workspace_guard, _clearvars_guard) = test_guard();
        ensure_test_resolver();
        let err = clearvars_builtin(vec![Value::Num(42.0)]).expect_err("expected type error");
        assert!(err
            .to_string()
            .contains("expected variable names as character vectors or string scalars"));
    }
}
