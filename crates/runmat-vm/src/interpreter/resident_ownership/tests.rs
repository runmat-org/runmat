use super::visibility::retained;
use super::*;
use runmat_accelerate_api::{handle_identity, GpuTensorHandle};
use runmat_runtime::context::{
    RuntimeCallRequest, RuntimeCallService, RuntimeContext, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn pure_host_arguments_do_not_register_roots_or_scan_global_owners() {
    let values = vec![Value::Tensor(
        runmat_value::Tensor::new(vec![2.0; 65_536], vec![256, 256]).unwrap(),
    )];
    let guard = ConsumedValues::new(&values);
    assert!(guard.allocations.is_none());
    assert!(guard.root.is_none());
    drop(guard);
}

#[test]
fn managed_only_pending_argument_exposes_later_residency_until_terminal_drop() {
    use std::future::Future;
    use std::task::{Context, Poll};

    for terminal in ["drop", "cancel", "error"] {
        runmat_gc::gc_test_context(|| {
            let owner = runmat_gc::gc_allocate_rooted(Value::Object(
                runmat_value::ObjectInstance::new("ManagedArgument"),
            ))
            .unwrap();
            let target = owner.handle();
            let argument = Value::HandleObject(runmat_value::HandleRef {
                class_name: "ManagedArgument".into(),
                target,
                valid: true,
            });
            let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), ()>>();
            let body = Box::pin(async move {
                let guard = ConsumedValues::new(std::slice::from_ref(&argument));
                assert!(
                    guard.root.is_some(),
                    "managed-only arguments need a live root"
                );
                assert!(guard.allocations.as_ref().unwrap().is_empty());
                drop(argument);
                receiver.await.unwrap()
            });
            let source = runmat_execution::cancellation::CancellationSource::new();
            let mut pending = Box::pin(futures::future::select(
                body,
                Box::pin(source.token().cancelled()),
            ));
            let mut cx = Context::from_waker(futures::task::noop_waker_ref());
            assert!(pending.as_mut().poll(&mut cx).is_pending());
            owner.unroot().unwrap();
            runmat_gc::gc_collect_major().unwrap();
            let resident = GpuTensorHandle::new(vec![1, 1], 9912, 9001);
            runmat_gc::gc_with_value_mut(&target, |value| {
                let Value::Object(object) = value else {
                    panic!("managed target")
                };
                object
                    .properties
                    .insert("resident".into(), Value::GpuTensor(resident.clone()));
            })
            .unwrap();
            let mut seen = HashSet::new();
            runmat_gc::gc_visit_retained_values(&[], &mut |value| retain_value(value, &mut seen))
                .unwrap();
            assert!(seen.contains(&handle_identity(&resident)));
            match terminal {
                "cancel" => {
                    source.cancel(runmat_execution::CancellationReason::User);
                    let Poll::Ready(futures::future::Either::Right((_, body))) =
                        pending.as_mut().poll(&mut cx)
                    else {
                        panic!("cancellation must win")
                    };
                    drop(body);
                }
                "error" => {
                    sender.send(Err(())).unwrap();
                    let Poll::Ready(futures::future::Either::Left((result, listener))) =
                        pending.as_mut().poll(&mut cx)
                    else {
                        panic!("error must complete")
                    };
                    assert_eq!(result, Err(()));
                    drop(listener);
                }
                "drop" => {}
                _ => unreachable!(),
            }
            drop(pending);
            let mut seen = HashSet::new();
            runmat_gc::gc_visit_retained_values(&[], &mut |value| retain_value(value, &mut seen))
                .unwrap();
            assert!(!seen.contains(&handle_identity(&resident)));
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
        });
    }
}

struct CapturingCall(RefCell<Vec<Value>>);
impl RuntimeCallService for CapturingCall {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for value in self.0.try_borrow().ok()?.iter() {
            visitor(value)?;
        }
        Some(())
    }
    fn resolve(&self, _: &str) -> Option<usize> {
        None
    }
    fn invoke(
        &self,
        _: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<runmat_value::ValueSequence, runmat_runtime::RuntimeError>>
    {
        unreachable!("root projection does not invoke callbacks")
    }
}
struct OpaqueCall;
impl RuntimeCallService for OpaqueCall {
    fn resolve(&self, _: &str) -> Option<usize> {
        None
    }
    fn invoke(
        &self,
        _: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<runmat_value::ValueSequence, runmat_runtime::RuntimeError>>
    {
        unreachable!("root projection does not invoke callbacks")
    }
}

#[test]
fn injected_callback_nested_alias_is_retained_and_borrow_failure_is_not_empty() {
    runmat_gc::gc_test_context(|| {
        let handle = GpuTensorHandle::new(vec![1, 1], 9123, 42);
        let nested = Value::Cell(
            runmat_value::CellArray::new_with_shape(
                vec![Value::GpuTensor(handle.clone())],
                vec![1, 1],
            )
            .unwrap(),
        );
        let calls = Rc::new(CapturingCall(RefCell::new(vec![nested])));
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_call(calls.clone()));
        futures::executor::block_on(runtime.scope(async {
            assert!(retained(&[], &HashSet::new())
                .unwrap()
                .contains(&handle_identity(&handle)));
            let _borrow = calls.0.borrow_mut();
            assert!(retained(&[], &HashSet::new())
                .unwrap_err()
                .contains("call service"));
        }));
    });
}

#[test]
fn opaque_callback_service_prevents_partial_alias_proof() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_call(Rc::new(OpaqueCall)));
        futures::executor::block_on(runtime.scope(async {
            assert!(retained(&[], &HashSet::new())
                .unwrap_err()
                .contains("call service"));
        }));
    });
}
