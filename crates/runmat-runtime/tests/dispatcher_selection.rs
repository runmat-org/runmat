//! Dispatcher behavior must link one runtime library, not a duplicate unit inventory.
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
use runmat_runtime::{
    build_runtime_error,
    builtin::{
        RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinImplementation,
        RuntimeBuiltinInvocation,
    },
    call_builtin_async_in_context,
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
    GpuGatherRetry, RuntimeError,
};
use runmat_value::{Value, ValueSequence};
use std::{cell::RefCell, error::Error, io, rc::Rc};

#[path = "support/retained_provider.rs"]
mod retained_provider;

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

fn fallback(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        record("fallback");
        let sequence = ValueSequence::single(Value::Num(17.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

fn fails_after_effect(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
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

fn untyped_failure(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        record("untyped");
        // Even conversion-like text is not an applicability capability.
        Err(RuntimeError::new("expected 1 args, got 0"))
    })
}

fn successful_override(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        record("override");
        let sequence = ValueSequence::single(Value::Num(29.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

fn sequence_override(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let requested_outputs = invocation.requested_outputs;
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
        let sequence = match requested_outputs {
            None => ValueSequence::single(Value::Num(7.0))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
            Some(0) => Ok(ValueSequence::empty()),
            Some(1) => ValueSequence::comma_separated(vec![Value::Num(11.0)])
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
            Some(count) => ValueSequence::comma_separated(
                (0..count).map(|index| Value::Num(index as f64)).collect(),
            )
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
        }?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

thread_local! {
    static INVOCATIONS: RefCell<Vec<(runmat_runtime::context::RuntimeContextLocalId, Option<usize>)>> =
        const { RefCell::new(Vec::new()) };
}

fn observe_invocation(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    let context = invocation
        .context
        .expect("session call carries its runtime");
    let outputs = invocation.requested_outputs;
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
        INVOCATIONS.with(|records| {
            records
                .borrow_mut()
                .push((context.local_identity(), outputs))
        });
        let sequence = ValueSequence::single(Value::Num(1.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
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
    let error = futures::executor::block_on(call_builtin_async_in_context(
        &runtime,
        "full",
        &[Value::Num(1.0)],
    ))
    .unwrap_err();
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
    let error = futures::executor::block_on(call_builtin_async_in_context(&runtime, "full", &[]))
        .unwrap_err();
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
        futures::executor::block_on(call_builtin_async_in_context(
            &runtime,
            "full",
            &[Value::Num(1.0)]
        ))
        .unwrap(),
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
        let error = futures::executor::block_on(call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::Num(1.0)],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
    }
    assert!(effects().is_empty());
}

#[test]
fn direct_embedding_call_does_not_inherit_entered_session_binding() {
    reset();
    let runtime = context(&[successful_override]);
    let _scope = runtime.enter();
    let result = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "full",
        &[Value::Num(1.0)],
    ))
    .expect("direct builtin call uses process inventory");
    assert_ne!(result, Value::Num(29.0));
    assert!(effects().is_empty());
}

#[test]
fn requested_gather_adapts_only_the_selected_implementation() {
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{AccelProvider, HostTensorView, ThreadProviderGuard};
    let provider = retained_provider::retained_test_provider!(PROVIDER: InProcessProvider = InProcessProvider::new());
    let _provider = ThreadProviderGuard::set(Some(provider));
    fn host_only(invocation: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
        let is_resident = matches!(invocation.args[0], Value::GpuTensor(_));
        runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async move {
            if is_resident {
                record("request-gather");
                Err(
                    build_runtime_error("host preparation requested before effect")
                        .with_gpu_gather_retry(GpuGatherRetry::Requested)
                        .build(),
                )
            } else {
                record("host-effect");
                let sequence = ValueSequence::single(Value::Num(31.0))
                    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
                runmat_runtime::execution::RootedValueSequence::new(sequence)
                    .map_err(|error| error.into_runtime_error())
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
        futures::executor::block_on(call_builtin_async_in_context(
            &runtime,
            "full",
            &[Value::GpuTensor(handle)]
        ))
        .unwrap(),
        Value::Num(31.0)
    );
    assert_eq!(effects(), ["request-gather", "host-effect"]);
}

#[test]
fn explicit_session_identity_and_output_request_reach_the_same_binding() {
    let first = context(&[observe_invocation]);
    let second = context(&[observe_invocation]);
    INVOCATIONS.with(|records| records.borrow_mut().clear());
    let first_call =
        runmat_runtime::call_builtin_async_in_context(&first, "full", &[Value::Num(1.0)]);
    let second_call = runmat_runtime::call_builtin_async_with_outputs_in_context(
        &second,
        "full",
        &[Value::Num(2.0)],
        1,
    );
    let (one, two) = futures::executor::block_on(async { futures::join!(first_call, second_call) });
    assert_eq!(one.unwrap(), Value::Num(1.0));
    assert_eq!(two.unwrap(), Value::Num(1.0));
    INVOCATIONS.with(|records| {
        assert_eq!(
            records.borrow().as_slice(),
            &[
                (first.local_identity(), None),
                (second.local_identity(), Some(1))
            ]
        );
    });
}

#[test]
fn canonical_binding_preserves_typed_default_empty_single_and_many_sequences() {
    let runtime = context(&[sequence_override]);
    let binding = RuntimeBuiltinBinding::new(
        BuiltinBindingIdentity {
            builtin: BuiltinCatalogIdentity { name: "full" },
            variant: "default",
        },
        sequence_override,
    );
    let cases = [
        (None, runmat_value::ValueSequenceKind::Single, 1),
        (Some(0), runmat_value::ValueSequenceKind::CommaSeparated, 0),
        (Some(1), runmat_value::ValueSequenceKind::CommaSeparated, 1),
        (Some(3), runmat_value::ValueSequenceKind::CommaSeparated, 3),
    ];
    for (requested_outputs, kind, length) in cases {
        let sequence = futures::executor::block_on((binding.implementation)(
            RuntimeBuiltinInvocation::new(&[], Some(runtime.clone()), requested_outputs),
        ))
        .expect("typed fixture sequence");
        assert_eq!(sequence.kind(), kind);
        assert_eq!(sequence.len(), length);
    }
}
