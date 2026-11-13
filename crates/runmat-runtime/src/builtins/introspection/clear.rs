//! MATLAB-compatible `clear` builtin for workspace variables.

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::{build_runtime_error, workspace, BuiltinResult};

#[runtime_builtin(
    name = "clear",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::clear"
)]
async fn clear_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if args.is_empty() {
        let detached = workspace::try_detach_selected(&workspace::WorkspaceDetachSelection::All)
            .map_err(clear_flow)?;
        cleanup_workspace_values(&detached).await?;
        return Ok(empty_return_value());
    }

    let mut names = Vec::new();
    for arg in &args {
        collect_clear_targets(arg, &mut names)?;
    }

    for name in names {
        if name.is_empty() {
            continue;
        }
        if let Some(value) = workspace::try_lookup(&name).map_err(clear_flow)? {
            drop(value);
            let detached =
                workspace::try_detach_selected(&workspace::WorkspaceDetachSelection::Names(vec![
                    name.clone(),
                ]))
                .map_err(clear_flow)?;
            cleanup_workspace_values(&detached).await?;
            continue;
        }
        match ClearDirective::decode(name) {
            ClearDirective::All => {
                let detached =
                    workspace::try_detach_selected(&workspace::WorkspaceDetachSelection::All)
                        .map_err(clear_flow)?;
                cleanup_workspace_values(&detached).await?;
                clear_dynamic_functions(crate::user_functions::DynamicFunctionClearRequest::All)
                    .await?;
            }
            ClearDirective::Variables => {
                let detached =
                    workspace::try_detach_selected(&workspace::WorkspaceDetachSelection::All)
                        .map_err(clear_flow)?;
                cleanup_workspace_values(&detached).await?;
            }
            ClearDirective::Functions => {
                clear_dynamic_functions(crate::user_functions::DynamicFunctionClearRequest::All)
                    .await?
            }
            ClearDirective::NativeExtensions => {
                clear_dynamic_functions(
                    crate::user_functions::DynamicFunctionClearRequest::NativeExtensions,
                )
                .await?
            }
            ClearDirective::Named(name) => {
                clear_dynamic_functions(crate::user_functions::DynamicFunctionClearRequest::Named(
                    name,
                ))
                .await?
            }
        }
    }

    Ok(empty_return_value())
}

enum ClearDirective {
    All,
    Variables,
    Functions,
    NativeExtensions,
    Named(String),
}

impl ClearDirective {
    fn decode(name: String) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "all" => Self::All,
            "variables" => Self::Variables,
            "functions" => Self::Functions,
            "mex" => Self::NativeExtensions,
            _ => Self::Named(name),
        }
    }
}

async fn clear_dynamic_functions(
    request: crate::user_functions::DynamicFunctionClearRequest,
) -> BuiltinResult<()> {
    match crate::user_functions::try_clear_dynamic_functions(request).await {
        Some(result) => result,
        None => Ok(()),
    }
}

async fn cleanup_workspace_values(
    detached: &workspace::DetachedWorkspaceValues,
) -> BuiltinResult<()> {
    crate::builtins::introspection::on_cleanup::run_cleanup_after_detach(detached).await
}

fn clear_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("clear").build()
}

fn clear_flow(mut error: crate::RuntimeError) -> crate::RuntimeError {
    error.context = error.context.with_builtin("clear");
    error
}

fn collect_clear_targets(arg: &Value, out: &mut Vec<String>) -> BuiltinResult<()> {
    match arg {
        Value::String(runmat_value::StringElement::Text(text)) => {
            out.push(text.trim().to_string());
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows > 1 {
                return Err(clear_error(
                    "clear: character array inputs must be a row vector or scalar text value",
                ));
            }
            out.push(
                super::text::decode_utf16_with_error(chars.data(), || {
                    clear_error("clear: character input is not valid UTF-16 text")
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
    use runmat_thread_local::runmat_thread_local;
    use runmat_value::Value;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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
        block_on(clear_builtin_async(args))
    }

    async fn clear_builtin_async(args: Vec<Value>) -> BuiltinResult<Value> {
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
            .scope(super::clear_builtin(
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
    fn variable_name_takes_precedence_over_clear_keyword() {
        let (_workspace_guard, _clear_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[("all", Value::Num(1.0)), ("kept", Value::Num(2.0))]);
        let requests = std::rc::Rc::new(RefCell::new(Vec::new()));
        let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let observed = std::rc::Rc::clone(&requests);
        context.set_dynamic_function_clearer(Some(std::rc::Rc::new(move |runtime, request| {
            let observed = std::rc::Rc::clone(&observed);
            crate::execution::admit_domain_continuation_with(&runtime, move || async move {
                observed.borrow_mut().push(request);
                Ok(())
            })
        })));
        let _context = crate::context::RuntimeContextGuard::enter(context);

        clear_builtin(vec![Value::from("all")]).expect("clear variable named all");

        assert!(requests.borrow().is_empty());
        let snapshot = crate::workspace::snapshot().unwrap_or_default();
        assert_eq!(snapshot, vec![("kept".into(), Value::Num(2.0))]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clear_mex_routes_only_to_native_extensions() {
        let (_workspace_guard, _clear_guard) = test_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let requests = std::rc::Rc::new(RefCell::new(Vec::new()));
        let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let observed = std::rc::Rc::clone(&requests);
        context.set_dynamic_function_clearer(Some(std::rc::Rc::new(move |runtime, request| {
            let observed = std::rc::Rc::clone(&observed);
            crate::execution::admit_domain_continuation_with(&runtime, move || async move {
                observed.borrow_mut().push(request);
                Ok(())
            })
        })));
        let _context = crate::context::RuntimeContextGuard::enter(context);

        clear_builtin(vec![Value::from("mex")]).expect("clear mex");

        assert_eq!(
            requests.borrow().as_slice(),
            &[crate::user_functions::DynamicFunctionClearRequest::NativeExtensions]
        );
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

    async fn clear_named_oncleanup_case() {
        let (_workspace_guard, _clear_guard) = test_guard();
        let _cleanup_guard = crate::builtins::introspection::on_cleanup::ON_CLEANUP_TEST_LOCK
            .lock()
            .unwrap();
        ensure_test_resolver();
        let (runtime, supervisor, _host) =
            crate::user_functions::test_support::exact_hosted_fixture_runtime();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 11);
        let counter = Arc::new(Mutex::new(0usize));
        let case = runtime.clone().scope(async {
            let _bound =
                crate::user_functions::test_support::install_bound_forwarder(binding.clone());
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                cleanup_invoker(Arc::clone(&counter)),
            ))
            .expect("call-port scope");
            let cleanup = crate::builtins::introspection::on_cleanup::on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            )
            .await
            .expect("create cleanup");
            cleanup.with_owned(|value| {
                set_workspace(&[("cleanup", value), ("x", Value::Num(1.0))]);
            });

            clear_builtin_async(vec![Value::from("cleanup")])
                .await
                .expect("clear cleanup");
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
        assert_eq!(snapshot[0].0, "x");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn clear_named_oncleanup_runs_callback_before_removal() {
        block_on(clear_named_oncleanup_case());
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test(async)]
    async fn clear_named_oncleanup_runs_callback_before_removal() {
        clear_named_oncleanup_case().await;
    }

    async fn clear_all_callback_rebinding_case() {
        let (_workspace_guard, _clear_guard) = test_guard();
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
                        crate::workspace::try_assign("new", Value::Num(9.0))?;
                        crate::workspace::try_assign("cleanup", Value::Num(2.0))?;
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
            cleanup.with_owned(|value| set_workspace(&[("cleanup", value)]));

            clear_builtin_async(vec![]).await.expect("clear all");
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
            vec![
                ("cleanup".into(), Value::Num(2.0)),
                ("new".into(), Value::Num(9.0)),
            ]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn clear_all_callback_rebinding_does_not_erase_new_workspace_values() {
        block_on(clear_all_callback_rebinding_case());
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test(async)]
    async fn clear_all_callback_rebinding_does_not_erase_new_workspace_values() {
        clear_all_callback_rebinding_case().await;
    }
}
