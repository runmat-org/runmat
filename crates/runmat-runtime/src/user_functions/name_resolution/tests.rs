use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
use std::{path::PathBuf, rc::Rc, sync::Arc};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dynamic_observation_places_only_scoped_private_semantic_before_native() {
    crate::builtins::wasm_registry::register_all();
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    runtime.set_native_name_observer(Some(Rc::new(|_| {
        Box::pin(async {
            Ok(Some(NativeNameFact {
                path: PathBuf::from("selected.mexa64"),
                origin: None,
                installed: true,
                installed_artifact: None,
            }))
        })
    })));
    let _entry = runtime.enter();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            matches!(name, "private_name" | "public_name").then_some(17)
        })))
        .expect("call-port scope");
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("observation must not invoke a candidate")
        })))
        .expect("call-port scope");
    let _scoped =
        crate::user_functions::install_scoped_name_target_resolver(Some(Rc::new(|name| {
            (name == "private_name").then_some(
                crate::user_functions::ScopedSemanticNameTarget::Published(
                    runmat_types::FunctionId(17),
                ),
            )
        })))
        .expect("call-port scope");
    let retired_owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let published_owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let _candidates =
        crate::user_functions::install_semantic_name_candidates(Some(Arc::new(move |name| {
            let fact = |path: &str, owner: Arc<runmat_value::FunctionDefinitionOwner>| {
                crate::user_functions::SemanticNameCandidate {
                    function: 17,
                    source: Some(SourceFunctionInfo {
                        source_id: runmat_types::SourceId(1),
                        name: name.into(),
                        function: 17,
                        role: runmat_types::FunctionSourceRole::Primary,
                        binding_origin: None,
                        source_file: Some(path.into()),
                        definition_owners: None,
                    }),
                    definition_owner: Some(owner),
                }
            };
            match name {
                "private_name" => vec![
                    fact("/old/private/private_name.m", Arc::clone(&retired_owner)),
                    fact("/current/private_name.m", Arc::clone(&published_owner)),
                ],
                "public_name" => vec![fact("/current/public_name.m", Arc::clone(&published_owner))],
                _ => Vec::new(),
            }
        })))
        .expect("call-port scope");
    let query = RuntimeNameQuery {
        route: RuntimeNameRoute::DynamicCall,
        filter: RuntimeNameFilter::Any,
        all: true,
    };
    let private = futures::executor::block_on(observe_runtime_name("private_name", query))
        .expect("private observation");
    assert!(matches!(
        private.selected(),
        Some(RuntimeNameCandidate::Semantic { function: 17, .. })
    ));
    assert!(matches!(
        private.shadowed().first(),
        Some(RuntimeNameCandidate::Native(_))
    ));
    let semantic_paths = private
        .candidates
        .iter()
        .filter_map(|candidate| match candidate {
            RuntimeNameCandidate::Semantic { source, .. } => {
                source.as_ref().and_then(|info| info.source_file.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        semantic_paths,
        vec!["/old/private/private_name.m", "/current/private_name.m"]
    );
    assert!(matches!(
        private.candidates.get(2),
        Some(RuntimeNameCandidate::Semantic { .. })
    ));
    let public = futures::executor::block_on(observe_runtime_name("public_name", query))
        .expect("public observation");
    assert!(matches!(
        public.selected(),
        Some(RuntimeNameCandidate::Native(_))
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn registered_constructor_precedes_native_and_source_but_builtin_collision_wins() {
    let temporary = tempfile::tempdir().expect("class collision root");
    std::fs::write(
        temporary.path().join("sin.m"),
        b"function y = sin(x)\n y=x;\nend",
    )
    .expect("source collision");
    std::fs::write(
        temporary.path().join("__registered_class_collision__.m"),
        b"function y = __registered_class_collision__()\n y=1;\nend",
    )
    .expect("source collision");
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_search_path(Arc::new(
        crate::builtins::common::path_state::SearchPath::new(
            temporary.path().to_string_lossy().into_owned(),
        ),
    ));
    runtime.set_native_name_observer(Some(Rc::new(|name| {
        Box::pin(async move {
            Ok(Some(NativeNameFact {
                path: PathBuf::from(format!("{name}.mexa64")),
                origin: None,
                installed: true,
                installed_artifact: None,
            }))
        })
    })));
    let _scope = crate::context::RuntimeContextGuard::enter(runtime);
    for name in ["sin", "__registered_class_collision__"] {
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: runmat_types::ClassIdentity::new(name).expect("class identity"),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
            events: HashMap::new(),
        });
    }
    let query = RuntimeNameQuery {
        route: RuntimeNameRoute::DynamicCall,
        filter: RuntimeNameFilter::Any,
        all: true,
    };
    let builtin =
        futures::executor::block_on(observe_runtime_name("sin", query)).expect("builtin collision");
    assert!(
        matches!(builtin.selected(), Some(RuntimeNameCandidate::Builtin(name)) if name.0 == "sin")
    );
    assert!(matches!(
        builtin.shadowed().first(),
        Some(RuntimeNameCandidate::RegisteredConstructor(_))
    ));
    assert!(builtin
        .candidates
        .iter()
        .any(|candidate| matches!(candidate, RuntimeNameCandidate::Native(_))));
    assert!(builtin
        .candidates
        .iter()
        .any(|candidate| matches!(candidate, RuntimeNameCandidate::Source(_))));
    let class = futures::executor::block_on(observe_runtime_name(
        "__registered_class_collision__",
        query,
    ))
    .expect("class collision");
    assert!(matches!(
        class.selected(),
        Some(RuntimeNameCandidate::RegisteredConstructor(_))
    ));
    assert!(matches!(
        class.shadowed().first(),
        Some(RuntimeNameCandidate::Native(_))
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn registered_static_method_is_later_than_native_and_source_name_routes() {
    let temporary = tempfile::tempdir().expect("static collision root");
    let package = temporary.path().join("+ObservedClass");
    std::fs::create_dir_all(&package).expect("package directory");
    std::fs::write(package.join("zero.m"), b"function y = zero()\n y=0;\nend")
        .expect("source collision");
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_search_path(Arc::new(
        crate::builtins::common::path_state::SearchPath::new(
            temporary.path().to_string_lossy().into_owned(),
        ),
    ));
    runtime.set_native_name_observer(Some(Rc::new(|_| {
        Box::pin(async {
            Ok(Some(NativeNameFact {
                path: PathBuf::from("ObservedClass.zero.mexa64"),
                origin: None,
                installed: true,
                installed_artifact: None,
            }))
        })
    })));
    let _scope = crate::context::RuntimeContextGuard::enter(runtime);
    let mut methods = HashMap::new();
    methods.insert(
        "zero".into(),
        crate::class_registry::RuntimeMethod {
            name: "zero".into(),
            is_static: true,
            is_abstract: false,
            is_sealed: false,
            access: runmat_types::MemberAccess::Public,
            function_name: "__observed_static_body__".to_owned(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: runmat_types::ClassIdentity::new("ObservedClass").expect("class identity"),
        parent: None,
        properties: HashMap::new(),
        methods,
        events: HashMap::new(),
    });
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "__observed_static_body__").then_some(55)
        })))
        .expect("call-port scope");
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("observation cannot invoke a static method")
        })))
        .expect("call-port scope");
    let resolution = futures::executor::block_on(observe_runtime_name(
        "ObservedClass.zero",
        RuntimeNameQuery {
            route: RuntimeNameRoute::DynamicCall,
            filter: RuntimeNameFilter::Any,
            all: true,
        },
    ))
    .expect("static collision");
    assert!(matches!(
        resolution.selected(),
        Some(RuntimeNameCandidate::Native(_))
    ));
    assert!(resolution
        .candidates
        .iter()
        .any(|candidate| matches!(candidate, RuntimeNameCandidate::Source(_))));
    assert!(resolution.candidates.iter().any(|candidate| matches!(
        candidate,
        RuntimeNameCandidate::RegisteredStaticMethod {
            resolution: crate::dispatcher::RegisteredStaticMethodResolution {
                target: crate::dispatcher::RegisteredStaticMethodTarget::Semantic(55),
                ..
            },
            ..
        }
    )));
    assert!(resolution.candidates.iter().any(|candidate| matches!(
        candidate,
        RuntimeNameCandidate::RegisteredStaticMethod { resolution, .. }
            if resolution.registration.method_name == runmat_types::MethodName::from("zero")
    )));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn semantic_candidates_keep_selected_and_shadowed_order() {
    crate::builtins::wasm_registry::register_all();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "sin").then_some(10)
        })))
        .expect("call-port scope");
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("observation cannot invoke a semantic function")
        })))
        .expect("call-port scope");
    let _observer =
        crate::user_functions::install_semantic_name_candidates(Some(Arc::new(|name| {
            if name == "sin" {
                [10, 11]
                    .into_iter()
                    .map(|function| crate::user_functions::SemanticNameCandidate {
                        function,
                        source: None,
                        definition_owner: None,
                    })
                    .collect()
            } else {
                Vec::new()
            }
        })))
        .expect("call-port scope");
    let result = futures::executor::block_on(observe_runtime_name(
        "sin",
        RuntimeNameQuery {
            route: RuntimeNameRoute::NamedFeval,
            filter: RuntimeNameFilter::Any,
            all: true,
        },
    ))
    .expect("ordered name facts");
    assert!(matches!(
        result.selected(),
        Some(RuntimeNameCandidate::Semantic { function: 10, .. })
    ));
    assert!(matches!(
        result.shadowed().first(),
        Some(RuntimeNameCandidate::Semantic { function: 11, .. })
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn resolver_without_invoker_never_selects_semantic_candidate() {
    crate::builtins::wasm_registry::register_all();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "sin").then_some(10)
        })))
        .expect("call-port scope");
    let _observer =
        crate::user_functions::install_semantic_name_candidates(Some(Arc::new(|name| {
            if name == "sin" {
                [10, 11]
                    .into_iter()
                    .map(|function| crate::user_functions::SemanticNameCandidate {
                        function,
                        source: None,
                        definition_owner: None,
                    })
                    .collect()
            } else {
                Vec::new()
            }
        })))
        .expect("call-port scope");
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(None).expect("call-port scope");
    for route in [RuntimeNameRoute::NamedFeval, RuntimeNameRoute::DynamicCall] {
        let result = futures::executor::block_on(observe_runtime_name(
            "sin",
            RuntimeNameQuery {
                route,
                filter: RuntimeNameFilter::Any,
                all: true,
            },
        ))
        .expect("resolver-only observation");
        assert!(result
            .candidates
            .iter()
            .all(|candidate| !matches!(candidate, RuntimeNameCandidate::Semantic { .. })));
        assert!(result
            .candidates
            .iter()
            .any(|candidate| matches!(candidate, RuntimeNameCandidate::Builtin(_))));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn changed_path_revision_never_publishes_stale_native_selection() {
    crate::builtins::wasm_registry::register_all();
    let path = Arc::new(crate::builtins::common::path_state::SearchPath::new(
        String::new(),
    ));
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_search_path(Arc::clone(&path));
    runtime.set_native_name_observer(Some(Rc::new(move |_| {
        crate::builtins::common::path_state::set_path_string("changed");
        Box::pin(async {
            Ok(Some(NativeNameFact {
                path: PathBuf::from("fixture.mexa64"),
                origin: None,
                installed: true,
                installed_artifact: None,
            }))
        })
    })));
    let _scope = crate::context::RuntimeContextGuard::enter(runtime);
    let result = futures::executor::block_on(observe_runtime_name(
        "fixture",
        RuntimeNameQuery {
            route: RuntimeNameRoute::DynamicCall,
            filter: RuntimeNameFilter::File,
            all: false,
        },
    ));
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("search path changed"));
    assert_eq!(path.generation(), 1);
}
