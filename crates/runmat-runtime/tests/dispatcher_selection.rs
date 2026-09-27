//! Dispatcher behavior must link one runtime library, not a duplicate unit inventory.
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
use runmat_runtime::{
    build_runtime_error,
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinImplementation},
    call_builtin,
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
    GpuGatherRetry, RuntimeError,
};
use runmat_value::Value;
use std::{cell::RefCell, error::Error, io, rc::Rc};

#[path = "dispatcher_selection/legacy.rs"]
mod legacy;

thread_local! {
    static EFFECTS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

fn record(effect: &'static str) {
    EFFECTS.with(|effects| effects.borrow_mut().push(effect));
}

fn reset() {
    EFFECTS.with(|effects| effects.borrow_mut().clear());
}

fn effects() -> Vec<&'static str> {
    EFFECTS.with(|effects| effects.borrow().clone())
}

fn fallback(_: &[Value]) -> RuntimeBuiltinFuture {
    Box::pin(async {
        record("fallback");
        Ok(Value::Num(17.0))
    })
}

fn fails_after_effect(_: &[Value]) -> RuntimeBuiltinFuture {
    Box::pin(async {
        record("selected");
        Err(
            build_runtime_error("selected invocation failed after its effect")
                .with_identifier("RunMat:Fixture:AfterEffect")
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .with_source(io::Error::new(io::ErrorKind::BrokenPipe, "original cause"))
                .build(),
        )
    })
}

fn untyped_failure(_: &[Value]) -> RuntimeBuiltinFuture {
    Box::pin(async {
        record("untyped");
        // Even conversion-like text is not an applicability capability.
        Err(RuntimeError::new("expected 1 args, got 0"))
    })
}

fn successful_override(_: &[Value]) -> RuntimeBuiltinFuture {
    Box::pin(async {
        record("override");
        Ok(Value::Num(29.0))
    })
}

struct Scoped(Vec<RuntimeBuiltinBinding>);
impl RuntimeBuiltinService for Scoped {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.0
            .iter()
            .copied()
            .filter(|binding| binding.identity.builtin.name == name)
            .collect()
    }
}

fn context(implementations: &[RuntimeBuiltinImplementation]) -> RuntimeContext {
    let bindings = implementations
        .iter()
        .copied()
        .map(|implementation| {
            RuntimeBuiltinBinding::new(
                BuiltinBindingIdentity {
                    builtin: BuiltinCatalogIdentity { name: "full" },
                    variant: "default",
                },
                implementation,
            )
        })
        .collect();
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Scoped(bindings))))
}

#[test]
fn selected_error_is_terminal_and_preserves_typed_cause() {
    reset();
    let runtime = context(&[fallback, fails_after_effect]);
    let _scope = runtime.enter();
    let error = call_builtin("full", &[Value::Num(1.0)]).unwrap_err();
    assert_eq!(effects(), ["selected"]);
    assert_eq!(error.identifier(), Some("RunMat:Fixture:AfterEffect"));
    assert_eq!(
        error.message(),
        "selected invocation failed after its effect"
    );
    assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
    let cause = error.source().unwrap().downcast_ref::<io::Error>().unwrap();
    assert_eq!(cause.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(cause.to_string(), "original cause");
}

#[test]
fn untyped_execution_error_is_not_an_overload_mismatch() {
    reset();
    let runtime = context(&[fallback, untyped_failure]);
    let _scope = runtime.enter();
    let error = call_builtin("full", &[]).unwrap_err();
    assert_eq!(effects(), ["untyped"]);
    assert_eq!(error.identifier(), None);
    assert_eq!(error.message(), "expected 1 args, got 0");
    assert!(error.source().is_none());
}

#[test]
fn successful_override_keeps_last_binding_priority() {
    reset();
    let runtime = context(&[fails_after_effect, successful_override]);
    let _scope = runtime.enter();
    assert_eq!(
        call_builtin("full", &[Value::Num(1.0)]).unwrap(),
        Value::Num(29.0)
    );
    assert_eq!(effects(), ["override"]);
}

#[test]
fn scoped_miss_does_not_consult_global_canonical_or_legacy_bindings() {
    reset();
    let runtime = context(&[]);
    let _scope = runtime.enter();
    for name in ["full", "mean", "__dispatcher_unregistered_fixture"] {
        let error = call_builtin(name, &[Value::Num(1.0)]).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
    }
    assert!(effects().is_empty());
}

#[test]
fn requested_gather_adapts_only_the_selected_implementation() {
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard};
    use std::sync::OnceLock;
    static PROVIDER: OnceLock<InProcessProvider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(InProcessProvider::new);
    let _provider = ThreadProviderGuard::set(Some(provider));
    fn host_only(args: &[Value]) -> RuntimeBuiltinFuture {
        let is_resident = matches!(args[0], Value::GpuTensor(_));
        Box::pin(async move {
            if is_resident {
                record("request-gather");
                Err(
                    build_runtime_error("host preparation requested before effect")
                        .with_gpu_gather_retry(GpuGatherRetry::Requested)
                        .build(),
                )
            } else {
                record("host-effect");
                Ok(Value::Num(31.0))
            }
        })
    }
    reset();
    let runtime = context(&[fallback, host_only]);
    let _scope = runtime.enter();
    let handle = provider
        .upload(&HostTensorView {
            data: &[3.0],
            shape: &[1, 1],
        })
        .unwrap()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    assert_eq!(
        call_builtin("full", &[Value::GpuTensor(handle)]).unwrap(),
        Value::Num(31.0)
    );
    assert_eq!(effects(), ["request-gather", "host-effect"]);
}
