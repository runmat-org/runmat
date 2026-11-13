//! MATLAB-compatible `clearvars` builtin for workspace variables.

use std::collections::HashSet;

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::{build_runtime_error, workspace, BuiltinResult};

#[runtime_builtin(
    name = "clearvars",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::clearvars"
)]
async fn clearvars_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if args.is_empty() {
        let detached = workspace::try_detach_selected(&workspace::WorkspaceDetachSelection::All)
            .map_err(clearvars_flow)?;
        cleanup_workspace_values(&detached).await?;
        return Ok(empty_return_value());
    }

    let mut words = Vec::new();
    for arg in &args {
        collect_clearvars_words(arg, &mut words)?;
    }

    let mut targets = Vec::new();
    let mut exclusions = Vec::new();
    let mut saw_except = false;
    for word in words {
        match ClearvarsWord::decode(word)? {
            ClearvarsWord::Empty => {}
            ClearvarsWord::Except => {
                if saw_except {
                    return Err(clearvars_error("clearvars: duplicate -except option"));
                }
                saw_except = true;
            }
            ClearvarsWord::Name(name) if saw_except => exclusions.push(name),
            ClearvarsWord::Name(name) => targets.push(name),
        }
    }

    if saw_except {
        if exclusions.is_empty() {
            return Err(clearvars_error(
                "clearvars: -except requires at least one variable name",
            ));
        }
        clear_except(&targets, &exclusions).await?;
    } else {
        let detached =
            workspace::try_detach_selected(&workspace::WorkspaceDetachSelection::Names(targets))
                .map_err(clearvars_flow)?;
        cleanup_workspace_values(&detached).await?;
    }

    Ok(empty_return_value())
}

enum ClearvarsWord {
    Empty,
    Except,
    Name(String),
}

impl ClearvarsWord {
    fn decode(word: String) -> BuiltinResult<Self> {
        let name = word.trim();
        if name.is_empty() {
            return Ok(Self::Empty);
        }
        if name.eq_ignore_ascii_case("-except") {
            return Ok(Self::Except);
        }
        if name.starts_with('-') {
            return Err(clearvars_error(format!(
                "clearvars: unsupported option '{name}'"
            )));
        }
        Ok(Self::Name(name.to_string()))
    }
}

async fn clear_except(targets: &[String], exclusions: &[String]) -> BuiltinResult<()> {
    let keep: HashSet<&str> = exclusions.iter().map(String::as_str).collect();
    if targets.is_empty() {
        let detached = workspace::try_detach_selected(
            &workspace::WorkspaceDetachSelection::Except(exclusions.to_vec()),
        )
        .map_err(clearvars_flow)?;
        cleanup_workspace_values(&detached).await?;
    } else {
        let names = targets
            .iter()
            .filter(|name| !keep.contains(name.as_str()))
            .cloned()
            .collect();
        let detached =
            workspace::try_detach_selected(&workspace::WorkspaceDetachSelection::Names(names))
                .map_err(clearvars_flow)?;
        cleanup_workspace_values(&detached).await?;
    }
    Ok(())
}

async fn cleanup_workspace_values(
    detached: &workspace::DetachedWorkspaceValues,
) -> BuiltinResult<()> {
    crate::builtins::introspection::on_cleanup::run_cleanup_after_detach(detached).await
}

fn clearvars_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("clearvars")
        .build()
}

fn clearvars_flow(mut error: crate::RuntimeError) -> crate::RuntimeError {
    error.context = error.context.with_builtin("clearvars");
    error
}

fn collect_clearvars_words(arg: &Value, out: &mut Vec<String>) -> BuiltinResult<()> {
    match arg {
        Value::String(runmat_value::StringElement::Text(text)) => {
            out.push(text.trim().to_string());
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows > 1 {
                return Err(clearvars_error(
                    "clearvars: character array inputs must be a row vector or scalar text value",
                ));
            }
            out.push(
                super::text::decode_utf16_with_error(chars.data(), || {
                    clearvars_error("clearvars: character input is not valid UTF-16 text")
                })?
                .trim()
                .to_string(),
            );
            Ok(())
        }
        Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
            for text in array.data() {
                out.push(
                    text.as_str()
                        .expect("checked nonmissing string")
                        .trim()
                        .to_string(),
                );
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
    use runmat_thread_local::runmat_thread_local;
    use runmat_value::Value;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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
        block_on(clearvars_builtin_async(args))
    }

    async fn clearvars_builtin_async(args: Vec<Value>) -> BuiltinResult<Value> {
        let owner = crate::workspace::BaseWorkspaceOwner::new();
        let mut delta = crate::workspace::WorkspaceDelta::new();
        TEST_WORKSPACE.with(|slot| {
            for (name, value) in slot.borrow().iter() {
                delta
                    .assign(name.clone(), value.clone())
                    .expect("root test value");
            }
        });
        owner.apply(delta).expect("seed exact workspace");
        let context = crate::context::legacy::active().unwrap_or_else(|| {
            crate::context::RuntimeContext::new(std::rc::Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ))
        });
        let ports = context
            .service_ports()
            .clone()
            .with_workspace(std::rc::Rc::new(
                crate::workspace::BaseWorkspaceService::new(owner.clone()),
            ));
        let context = context.with_service_ports(ports);
        let result = context
            .scope(super::clearvars_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .await;
        let remaining = owner.snapshot().expect("read exact workspace").to_values();
        TEST_WORKSPACE.with(|slot| *slot.borrow_mut() = remaining.into_iter().collect());
        result
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

    fn cleanup_invoker(
        counter: Arc<Mutex<usize>>,
    ) -> Arc<crate::user_functions::UnscopedFunctionInvoker> {
        Arc::new(move |_function, args, requested_outputs| {
            assert!(
                args.is_empty(),
                "onCleanup callback should receive no inputs"
            );
            assert_eq!(requested_outputs, 0);
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                *counter.lock().unwrap() += 1;
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    runmat_value::Tensor::zeros(vec![0, 0]),
                ))
            })
        })
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

    async fn clearvars_except_cleanup_case() {
        let (_workspace_guard, _clearvars_guard) = test_guard();
        let _cleanup_guard = crate::builtins::introspection::on_cleanup::ON_CLEANUP_TEST_LOCK
            .lock()
            .unwrap();
        ensure_test_resolver();
        let (runtime, supervisor, _host) =
            crate::user_functions::test_support::exact_hosted_fixture_runtime();
        let removed_binding =
            crate::user_functions::test_support::exact_local_binding(&runtime, 11);
        let kept_binding = crate::user_functions::test_support::exact_local_binding(&runtime, 12);
        let counter = Arc::new(Mutex::new(0usize));
        let case = runtime.clone().scope(async {
            let _bound = crate::user_functions::test_support::install_bound_forwarders(vec![
                removed_binding.clone(),
                kept_binding.clone(),
            ]);
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                cleanup_invoker(Arc::clone(&counter)),
            ))
            .expect("call-port scope");
            let removed = crate::builtins::introspection::on_cleanup::on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", removed_binding),
            )
            .await
            .expect("create removed cleanup");
            let kept = crate::builtins::introspection::on_cleanup::on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", kept_binding),
            )
            .await
            .expect("create kept cleanup");
            removed.with_owned(|removed_value| {
                kept.with_owned(|kept_value| {
                    set_workspace(&[("removed", removed_value), ("kept", kept_value)]);
                });
            });

            clearvars_builtin_async(vec![Value::from("-except"), Value::from("kept")])
                .await
                .expect("clearvars");
        });
        #[cfg(not(target_arch = "wasm32"))]
        supervisor
            .participate_foreground(case)
            .await
            .expect("drive cleanup");
        #[cfg(target_arch = "wasm32")]
        {
            let _supervisor = supervisor;
            case.await;
        }

        assert_eq!(*counter.lock().unwrap(), 1);
        let snapshot = crate::workspace::snapshot().unwrap_or_default();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].0, "kept");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn clearvars_except_runs_cleanup_for_removed_values_only() {
        block_on(clearvars_except_cleanup_case());
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test(async)]
    async fn clearvars_except_runs_cleanup_for_removed_values_only() {
        clearvars_except_cleanup_case().await;
    }

    async fn clearvars_named_batch_rebinding_case() {
        let (_workspace_guard, _clearvars_guard) = test_guard();
        let _cleanup_guard = crate::builtins::introspection::on_cleanup::ON_CLEANUP_TEST_LOCK
            .lock()
            .unwrap();
        ensure_test_resolver();
        let (runtime, supervisor, _host) =
            crate::user_functions::test_support::exact_hosted_fixture_runtime();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 11);
        let case = runtime.clone().scope(async {
            let _bound =
                crate::user_functions::test_support::install_bound_forwarder(binding.clone());
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(|_function, _, _| {
                    Box::pin(async {
                        crate::workspace::try_assign("y", Value::Num(2.0))?;
                        let output = crate::sequence::single_value_sequence(Value::Num(0.0))?;
                        crate::execution::RootedValueSequence::new(output)
                            .map_err(|error| error.into_runtime_error())
                    })
                }),
            ))
            .expect("call-port scope");
            let cleanup = crate::builtins::introspection::on_cleanup::on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            )
            .await
            .expect("create cleanup");
            cleanup.with_owned(|value| {
                set_workspace(&[("x", value), ("y", Value::Num(1.0))]);
            });

            clearvars_builtin_async(vec![Value::from("x"), Value::from("y")])
                .await
                .expect("clearvars");
        });
        #[cfg(not(target_arch = "wasm32"))]
        supervisor
            .participate_foreground(case)
            .await
            .expect("drive cleanup");
        #[cfg(target_arch = "wasm32")]
        {
            let _supervisor = supervisor;
            case.await;
        }

        assert_eq!(
            crate::workspace::snapshot().unwrap_or_default(),
            vec![("y".into(), Value::Num(2.0))]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn clearvars_named_batch_does_not_erase_callback_rebinding() {
        block_on(clearvars_named_batch_rebinding_case());
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test(async)]
    async fn clearvars_named_batch_does_not_erase_callback_rebinding() {
        clearvars_named_batch_rebinding_case().await;
    }
}
