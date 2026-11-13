use super::*;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
use runmat_runtime::builtin::{
    RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation,
};
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeServicePorts};
use runmat_value::ValueSequence;
use std::cell::RefCell;
use std::rc::Rc;

fn runtime() -> runmat_runtime::context::RuntimeContext {
    runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

const DIRECT: &str = "__vm_import_once";
const EXPLICIT: &str = "PackageA.__vm_import_once";
const OTHER: &str = "PackageB.__vm_import_once";

thread_local! {
    static BODY_CALLS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

struct ScopedBindings(Vec<RuntimeBuiltinBinding>);

impl RuntimeBuiltinService for ScopedBindings {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.0
            .iter()
            .copied()
            .filter(|binding| binding.identity.builtin.name == name)
            .collect()
    }
}

fn scoped_runtime(bindings: Vec<RuntimeBuiltinBinding>) -> runmat_runtime::context::RuntimeContext {
    runtime().with_service_ports(
        RuntimeServicePorts::default().with_builtin(Rc::new(ScopedBindings(bindings))),
    )
}

fn binding(
    name: &'static str,
    implementation: runmat_runtime::builtin::RuntimeBuiltinImplementation,
) -> RuntimeBuiltinBinding {
    RuntimeBuiltinBinding::new(
        BuiltinBindingIdentity {
            builtin: BuiltinCatalogIdentity { name },
            variant: "default",
        },
        implementation,
    )
}

fn fixture_body(name: &'static str, value: f64) -> RuntimeBuiltinFuture {
    BODY_CALLS.with_borrow_mut(|calls| calls.push(name));
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
        let sequence = ValueSequence::single(Value::Num(value))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

fn direct_body(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    fixture_body(DIRECT, 1.0)
}

fn explicit_body(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    fixture_body(EXPLICIT, 2.0)
}

fn other_body(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    fixture_body(OTHER, 3.0)
}

fn failing_body(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    BODY_CALLS.with_borrow_mut(|calls| calls.push(EXPLICIT));
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        Err(
            runmat_runtime::build_runtime_error("selected import failed")
                .with_identifier("RunMat:ImportFixtureFailure")
                .build(),
        )
    })
}

fn undefined_body(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    BODY_CALLS.with_borrow_mut(|calls| calls.push(DIRECT));
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        Err(
            runmat_runtime::build_runtime_error("body-defined undefined function")
                .with_identifier("RunMat:UndefinedFunction")
                .build(),
        )
    })
}

fn imports() -> Vec<(Vec<String>, bool)> {
    vec![
        (vec!["PackageA".into(), DIRECT.into()], false),
        (vec!["PackageB".into(), DIRECT.into()], false),
    ]
}

#[test]
fn direct_success_never_executes_imported_body() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            BODY_CALLS.with_borrow_mut(Vec::clear);
            let runtime = scoped_runtime(vec![
                binding(DIRECT, direct_body),
                binding(EXPLICIT, explicit_body),
            ]);
            let _scope = runtime.enter();
            let value = call_builtin_or_imported(&runtime, DIRECT, &imports(), &[], 1)
                .await
                .expect("direct body wins");
            assert_eq!(value.as_slice(), &[Value::Num(1.0)]);
            BODY_CALLS.with_borrow(|calls| assert_eq!(calls.as_slice(), &[DIRECT]));
        });
    });
}

#[test]
fn selected_direct_body_error_does_not_authorize_import_fallback() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            BODY_CALLS.with_borrow_mut(Vec::clear);
            let runtime = scoped_runtime(vec![
                binding(DIRECT, undefined_body),
                binding(EXPLICIT, explicit_body),
            ]);
            let _scope = runtime.enter();
            let error = call_builtin_or_imported(&runtime, DIRECT, &imports(), &[], 1)
                .await
                .expect_err("selected direct body failure is terminal");
            assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
            BODY_CALLS.with_borrow(|calls| assert_eq!(calls.as_slice(), &[DIRECT]));
        });
    });
}

#[test]
fn ambiguous_imports_execute_no_body() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            BODY_CALLS.with_borrow_mut(Vec::clear);
            let runtime = scoped_runtime(vec![
                binding(EXPLICIT, explicit_body),
                binding(OTHER, other_body),
            ]);
            let _scope = runtime.enter();
            let error = call_builtin_or_imported(&runtime, DIRECT, &imports(), &[], 1)
                .await
                .expect_err("two qualified targets are ambiguous");
            assert_eq!(error.identifier(), Some("RunMat:AmbiguousBuiltinImport"));
            BODY_CALLS.with_borrow(|calls| assert!(calls.is_empty()));
        });
    });
}

#[test]
fn ambiguous_wildcard_imports_execute_no_body() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            BODY_CALLS.with_borrow_mut(Vec::clear);
            let runtime = scoped_runtime(vec![
                binding(EXPLICIT, explicit_body),
                binding(OTHER, other_body),
            ]);
            let _scope = runtime.enter();
            let imports = vec![
                (vec!["PackageA".into()], true),
                (vec!["PackageB".into()], true),
            ];
            let error = call_builtin_or_imported(&runtime, DIRECT, &imports, &[], 1)
                .await
                .expect_err("two wildcard targets are ambiguous");
            assert_eq!(error.identifier(), Some("RunMat:AmbiguousBuiltinImport"));
            BODY_CALLS.with_borrow(|calls| assert!(calls.is_empty()));
        });
    });
}

#[test]
fn selected_import_failure_does_not_try_wildcard_or_another_body() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            BODY_CALLS.with_borrow_mut(Vec::clear);
            let runtime = scoped_runtime(vec![
                binding(EXPLICIT, failing_body),
                binding(OTHER, other_body),
            ]);
            let _scope = runtime.enter();
            let imports = vec![
                (vec!["PackageA".into(), DIRECT.into()], false),
                (vec!["PackageB".into()], true),
            ];
            let error = call_builtin_or_imported(&runtime, DIRECT, &imports, &[], 1)
                .await
                .expect_err("selected failure propagates");
            assert_eq!(error.identifier(), Some("RunMat:ImportFixtureFailure"));
            BODY_CALLS.with_borrow(|calls| assert_eq!(calls.as_slice(), &[EXPLICIT]));
        });
    });
}

#[test]
fn duplicate_import_identity_executes_once() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            BODY_CALLS.with_borrow_mut(Vec::clear);
            let runtime = scoped_runtime(vec![binding(EXPLICIT, explicit_body)]);
            let _scope = runtime.enter();
            let imports = vec![
                (vec!["PackageA".into(), DIRECT.into()], false),
                (vec!["PackageA".into(), DIRECT.into()], false),
            ];
            let value = call_builtin_or_imported(&runtime, DIRECT, &imports, &[], 1)
                .await
                .expect("duplicate identity is one candidate");
            assert_eq!(value.as_slice(), &[Value::Num(2.0)]);
            BODY_CALLS.with_borrow(|calls| assert_eq!(calls.as_slice(), &[EXPLICIT]));
        });
    });
}

#[test]
fn direct_dispatch_keeps_outputs_and_runtime_errors() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            let runtime = runtime();
            let _scope = runtime.enter();
            let args = [Value::Tensor(
                runmat_value::Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            )];
            let result = runmat_runtime::call_builtin_async_with_outputs("size", &args, 2)
                .await
                .unwrap();
            let Value::OutputList(outputs) = result.value() else {
                panic!("two outputs")
            };
            assert_eq!(outputs.as_slice(), &[Value::Num(2.), Value::Num(3.)]);
            let error = runmat_runtime::call_builtin_async_with_outputs(
                "__missing_observed_builtin__",
                &[],
                1,
            )
            .await
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
        });
    });
}

#[test]
fn imported_call_uses_runtime_dispatch_with_original_arguments() {
    runmat_gc::gc_test_context(|| {
        futures::executor::block_on(async {
            let runtime = runtime();
            let _scope = runtime.enter();
            let args = [Value::Tensor(
                runmat_value::Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            )];
            let result = crate::call::builtins::resolve_imported_builtin(
                &runtime,
                "numel",
                &[(vec!["numel".into()], false)],
                &args,
                1,
            )
            .await
            .unwrap();
            let crate::call::builtins::ImportedBuiltinResolution::Resolved(value) = result else {
                panic!("resolved import")
            };
            assert_eq!(value.kind(), runmat_value::ValueSequenceKind::Single);
            assert_eq!(value.as_slice(), [Value::Num(6.)]);
        });
    });
}
