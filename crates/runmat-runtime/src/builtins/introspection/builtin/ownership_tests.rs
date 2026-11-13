use super::*;
use crate::builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation};
use crate::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use std::{cell::Cell, rc::Rc};

thread_local! {
    static ORIGINAL: Cell<usize> = const { Cell::new(0) };
}

struct Registered;
impl RuntimeBuiltinService for Registered {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        if name != "mean" {
            return Vec::new();
        }
        vec![RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::BuiltinCatalogIdentity { name: "mean" },
                variant: "default",
            },
            selected,
        )]
    }
}

fn selected(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let input = invocation.into_continuation_input();
    crate::builtin::admitted_builtin_future(None, "fixture", async move {
        let input = input?;
        assert!(
            input.context.is_some(),
            "retain explicit invocation context"
        );
        assert_eq!(input.requested_outputs, Some(1));
        ORIGINAL.with(|pointer| {
            assert_eq!(
                input.args.get(0).unwrap() as *const Value as usize,
                pointer.get()
            );
        });
        let Value::HandleObject(handle) = &input.args[1] else {
            panic!("original captured GC edge");
        };
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle.target, |_| ()).is_ok());
        crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(7.))
    })
}

#[test]
fn builtin_bypasses_methods_and_constructors_without_republishing_captures() {
    runmat_gc::gc_test_context(|| {
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_builtin(Rc::new(Registered)),
                );
        let _entry = runtime.enter();
        let class = "BuiltinOriginalCaptureBypassFixture";
        let mut methods = std::collections::HashMap::new();
        methods.insert(
            "mean".into(),
            crate::class_registry::RuntimeMethod {
                name: "mean".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "builtin_bypass_fixture_method".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: class.into(),
            parent: None,
            properties: Default::default(),
            methods,
            events: Default::default(),
        });
        let _resolver = crate::user_functions::install_semantic_function_resolver(Some(
            std::sync::Arc::new(|name| (name == "builtin_bypass_fixture_method").then_some(29022)),
        ))
        .unwrap();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(|function, _, _| {
                assert_eq!(function, 29022);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        9.,
                    ))
                })
            }),
        ))
        .unwrap();
        let root = runmat_gc::gc_allocate_rooted(Value::Num(43.)).unwrap();
        let handle = root.handle();
        let source = crate::execution::RetainedValueReadLease::prepare_owned(
            Value::Closure(runmat_value::Closure::new(
                "builtin_original_captures".into(),
                None,
                runmat_types::FunctionSourceRole::Anonymous,
                None,
                vec![
                    Value::Object(runmat_value::ObjectInstance::new(class)),
                    Value::HandleObject(runmat_value::HandleRef {
                        class_name: "BuiltinCapturedEdge".into(),
                        target: handle,
                        valid: true,
                    }),
                ],
                None,
            )),
            &runtime,
        )
        .unwrap();
        let arguments = crate::call::BuiltinArguments::new(
            crate::call::RootedFunctionArguments::captured(
                source,
                crate::execution::RootedValues::new(Vec::new()).unwrap(),
            )
            .unwrap(),
        );
        ORIGINAL.with(|pointer| pointer.set(arguments.get(0).unwrap() as *const Value as usize));
        root.unroot().unwrap();
        let normal = futures::executor::block_on(
            crate::dispatcher::try_call_retained_builtin_sequence_with_outputs(
                "mean",
                arguments.fork(),
                1,
                Some(&runtime),
            ),
        )
        .unwrap()
        .unwrap();
        assert_eq!(normal.value().as_slice(), &[Value::Num(9.)]);
        let bypass = futures::executor::block_on(builtin_builtin(
            Some(runtime.clone()),
            "mean".into(),
            arguments,
        ))
        .unwrap();
        assert_eq!(bypass, Value::Num(7.));
        drop(normal);
        drop(bypass);
        let empty = crate::execution::RootedValues::new(Vec::new()).unwrap();
        let constructor = futures::executor::block_on(
            crate::dispatcher::try_call_retained_builtin_sequence_with_outputs(
                class,
                empty.fork().into(),
                1,
                Some(&runtime),
            ),
        )
        .unwrap()
        .expect("normal dispatch still admits the registered constructor");
        assert!(matches!(constructor.value().as_slice(), [Value::Object(_)]));
        let error =
            futures::executor::block_on(builtin_builtin(Some(runtime), class.into(), empty.into()))
                .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
    });
}
