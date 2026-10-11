use super::*;
use std::{collections::HashMap, path::PathBuf, rc::Rc, sync::Arc};

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
        })));
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("observation cannot invoke a static method")
        })));
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
        })));
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("observation cannot invoke a semantic function")
        })));
    let _observer =
        crate::user_functions::install_semantic_name_candidates(Some(Arc::new(|name| {
            if name == "sin" {
                vec![10, 11]
            } else {
                Vec::new()
            }
        })));
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
        })));
    let _observer =
        crate::user_functions::install_semantic_name_candidates(Some(Arc::new(|name| {
            if name == "sin" {
                vec![10, 11]
            } else {
                Vec::new()
            }
        })));
    let _invoker = crate::user_functions::install_semantic_function_invoker(None);
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
