//! Query authority follows the selected binding, never a requested spelling.
use super::*;
use crate::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};

struct Alias {
    requested: &'static str,
    binding: crate::builtin::RuntimeBuiltinBinding,
}
impl RuntimeBuiltinService for Alias {
    fn bindings_by_name(&self, name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
        if name == self.requested {
            vec![self.binding]
        } else {
            Vec::new()
        }
    }
}

fn canonical(name: &str) -> crate::builtin::RuntimeBuiltinBinding {
    let selected = select_registered_builtin(name, None).unwrap().unwrap();
    let SelectedImplementation::Canonical(binding) = selected.implementation else {
        panic!("fixture requires the actual canonical binding");
    };
    binding
}

#[test]
fn validity_query_uses_selected_binding_for_execution_and_observation() {
    runmat_gc::gc_test_context(|| {
        crate::dispatcher::ensure_wasm_builtins_registered();
        let query = canonical("isvalid");
        let ordinary = canonical("sin");
        let root = runmat_gc::gc_allocate_rooted(Value::Object(runmat_value::ObjectInstance::new(
            "ExactValidityFixture",
        )))
        .unwrap();
        for valid in [true, false] {
            let receiver = Value::HandleObject(runmat_value::HandleRef {
                class_name: "ExactValidityFixture".into(),
                target: root.handle(),
                valid,
            });
            let runtime =
                RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(
                        Alias {
                            requested: "selected_query_alias",
                            binding: query,
                        },
                    )));
            let result = futures::executor::block_on(crate::call_builtin_async_in_context(
                &runtime,
                "selected_query_alias",
                std::slice::from_ref(&receiver),
            ))
            .unwrap();
            assert_eq!(result.value(), &Value::Bool(valid));
            assert!(matches!(observe_builtin_dispatch_target(
                &runtime, "selected_query_alias", std::slice::from_ref(&receiver),
            ).unwrap(), Some(ObservedBuiltinDispatchTarget::Builtin(identity))
                if identity == runmat_types::BuiltinId("isvalid".into())));
            // The exact-binding direct path must consume the same catalog role.
            let arguments = crate::call::BuiltinArguments::from_borrowed(
                std::slice::from_ref(&receiver),
                Some(&runtime),
            )
            .unwrap();
            let result = futures::executor::block_on(
                crate::dispatcher::call_selected_retained_builtin_async_with_outputs(
                    query,
                    arguments,
                    1,
                    Some(runtime.clone()),
                ),
            )
            .unwrap();
            assert_eq!(result.as_slice(), &[Value::Bool(valid)]);
            if !valid {
                let ordinary_error = select_instance_method(
                    "move",
                    crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&receiver)),
                )
                .err()
                .expect("ordinary method still rejects invalid receiver");
                assert_eq!(
                    ordinary_error.identifier(),
                    Some("RunMat:SelectedObjectReceiverInvalid")
                );
                let shadow =
                    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(
                            Alias {
                                requested: "isvalid",
                                binding: ordinary,
                            },
                        )));
                let error = observe_builtin_dispatch_target(
                    &shadow,
                    "isvalid",
                    std::slice::from_ref(&receiver),
                )
                .expect_err("requested query spelling grants no authority");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:SelectedObjectReceiverInvalid")
                );
                let error = futures::executor::block_on(crate::call_builtin_async_in_context(
                    &shadow,
                    "isvalid",
                    std::slice::from_ref(&receiver),
                ))
                .unwrap_err();
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:SelectedObjectReceiverInvalid")
                );
            }
        }
        // A real checked value-class method keeps its exact body binding.
        // The handle-only sealed role must not swallow this ordinary overload.
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let class = runmat_types::ClassIdentity::from("ValueValidityOverloadFixture");
        let method = runmat_types::MethodName::from("isvalid");
        let binding = crate::user_functions::test_support::exact_portable_binding(
            1773,
            "value-class validity overload body",
        );
        let _catalog = crate::user_functions::install_source_function_catalog(Some(
            std::sync::Arc::new(vec![crate::user_functions::SourceFunctionInfo {
                source_id: runmat_types::SourceId(1),
                name: "value_validity_body".into(),
                function: binding.function,
                role: runmat_types::FunctionSourceRole::Method,
                binding_origin: binding.authority.portable_origin(),
                source_file: None,
                definition_owners: None,
            }]),
        ))
        .unwrap();
        crate::class_registry::register_checked_classes(vec![
            crate::class_registry::CheckedClassRegistration {
                class: crate::class_registry::RuntimeClass {
                    name: class.clone(),
                    parent: None,
                    properties: Default::default(),
                    methods: std::collections::HashMap::from([(
                        method.clone(),
                        crate::class_registry::RuntimeMethod {
                            name: method.clone(),
                            is_static: false,
                            is_abstract: false,
                            is_sealed: false,
                            access: runmat_types::MemberAccess::Public,
                            function_name: "value_validity_body".into(),
                            implicit_class_argument: None,
                            role: runmat_types::ClassMethodRole::Ordinary,
                        },
                    )]),
                    events: Default::default(),
                },
                method_bindings: std::collections::HashMap::from([(
                    method.clone(),
                    binding.clone(),
                )]),
                is_sealed: false,
                is_abstract: false,
                enumerations: Vec::new(),
            },
        ])
        .unwrap();
        let _forward = crate::user_functions::test_support::install_bound_forwarder(binding);
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = calls.clone();
        let expected_class = class.clone();
        let _invoke = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(move |function, arguments, outputs| {
                assert_eq!(function, 1773);
                assert_eq!(outputs, 1);
                assert_eq!(arguments.len(), 1);
                assert!(matches!(&arguments[0], Value::Object(object)
                    if object.class_name() == &expected_class));
                observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Bool(
                        true,
                    ))
                })
            }),
        ))
        .unwrap();
        let receiver = Value::Object(runmat_value::ObjectInstance::new(class.clone()));
        let result = futures::executor::block_on(crate::call_builtin_async_in_context(
            &runtime,
            "isvalid",
            std::slice::from_ref(&receiver),
        ))
        .unwrap();
        assert_eq!(result.value(), &Value::Bool(true));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let arguments = crate::call::BuiltinArguments::from_borrowed(
            std::slice::from_ref(&receiver),
            Some(&runtime),
        )
        .unwrap();
        let result = futures::executor::block_on(
            crate::dispatcher::call_selected_retained_builtin_async_with_outputs(
                query,
                arguments,
                1,
                Some(runtime.clone()),
            ),
        )
        .unwrap();
        assert_eq!(result.as_slice(), &[Value::Bool(true)]);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(matches!(observe_builtin_dispatch_target(
            &runtime, "isvalid", std::slice::from_ref(&receiver),
        ).unwrap(), Some(ObservedBuiltinDispatchTarget::InstanceMethod { owner, method: selected })
            if owner == class && selected == method));
    });
}
