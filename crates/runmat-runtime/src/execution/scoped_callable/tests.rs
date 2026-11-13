use super::*;
use crate::execution::RootedValues;
use futures::{executor::block_on, task::noop_waker_ref};
use runmat_value::{Value, ValueSequence};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    task::{Context, Poll},
};

struct Managed(crate::context::host_effect::ManagedHostEffectLedger);
impl crate::context::RuntimeParallelService for Managed {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        crate::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(
        &self,
    ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[test]
fn finite_creation_and_same_issuer_finite_invocation_refuse_before_user_effects() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(1 << 20);
    let managed = runtime.clone().with_service_ports(
        runtime
            .service_ports()
            .clone()
            .with_parallel(Rc::new(Managed(ledger))),
    );
    let token = runtime.cancellation();
    let mut made = 0;
    let error = block_on(drive(
        Some(&managed),
        Some(&token),
        |_, _| {
            made += 1;
            Ok(std::future::ready(returned()))
        },
        async |_| Ok(ControlFlow::<(), _>::Continue(returned()?)),
    ))
    .unwrap_err();
    assert_eq!(made, 0);
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    let mut calls = 0;
    let result = block_on(drive(
        Some(&runtime),
        Some(&token),
        |handle, original| {
            let error = validate_target(Some(&managed), &handle).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:Placement:InsufficientResourceEvidence")
            );
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            Ok(async move {
                invoke(
                    original.as_ref(),
                    handle,
                    arguments(original.as_ref()),
                    1,
                    None,
                )?
                .await
            })
        },
        async |_| {
            calls += 1;
            Ok(ControlFlow::<(), _>::Continue(returned()?))
        },
    ))
    .unwrap();
    assert!(matches!(result, ControlFlow::Continue(_)));
    assert_eq!(calls, 1);
}

fn arguments(runtime: Option<&RuntimeContext>) -> RootedFunctionArguments {
    let values = vec![Value::Num(3.0)];
    let values = match runtime {
        Some(runtime) => RootedValues::in_context(values, runtime).unwrap(),
        None => RootedValues::new(values).unwrap(),
    };
    RootedFunctionArguments::ordinary(values)
}
fn returned() -> BuiltinResult<RootedValueSequence> {
    RootedValueSequence::new(
        ValueSequence::single(Value::Num(9.0))
            .map_err(crate::sequence::sequence_error_to_runtime)?,
    )
    .map_err(|error| error.into_runtime_error())
}

#[test]
fn nested_requests_borrow_one_handler_and_expired_foreign_and_busy_ports_fail_before_effects() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let token = runtime.cancellation();
    let retained = Rc::new(RefCell::new(None::<ScopedFunctionHandle>));
    let issued = retained.clone();
    let active = retained.clone();
    let mut actual_calls = 0;
    let result = block_on(drive(
        Some(&runtime),
        Some(&token),
        |handle, original| {
            *issued.borrow_mut() = Some(handle.clone());
            let error = invoke(
                Some(&foreign),
                handle.clone(),
                arguments(Some(&foreign)),
                1,
                None,
            )
            .err()
            .expect("foreign refusal");
            assert_eq!(error.identifier(), Some("RunMat:ScopedCallable:Foreign"));
            Ok(async move {
                invoke(
                    original.as_ref(),
                    handle,
                    arguments(original.as_ref()),
                    1,
                    None,
                )?
                .await
            })
        },
        async |request| {
            actual_calls += 1;
            assert_eq!(request.requested_outputs, 1);
            assert_eq!(request.arguments.view().get(0), Some(&Value::Num(3.0)));
            let error = invoke(
                Some(&runtime),
                active.borrow().as_ref().unwrap().clone(),
                arguments(Some(&runtime)),
                1,
                None,
            )
            .err()
            .expect("recursive refusal");
            assert_eq!(error.identifier(), Some("RunMat:ScopedCallable:Busy"));
            Ok(ControlFlow::<(), _>::Continue(returned()?))
        },
    ))
    .unwrap();
    let ControlFlow::Continue(result) = result else {
        panic!("actual child result")
    };
    assert_eq!(result.as_slice(), &[Value::Num(9.0)]);
    assert_eq!(actual_calls, 1);
    let handle = retained.borrow().as_ref().unwrap().clone();
    assert!(!handle.is_live());
    let error = invoke(Some(&runtime), handle, arguments(Some(&runtime)), 1, None)
        .err()
        .expect("expired refusal");
    assert_eq!(error.identifier(), Some("RunMat:ScopedCallable:Expired"));
}

#[test]
fn original_handler_failure_is_not_cloned_replayed_or_reclassified() {
    let mut actual_calls = 0;
    let error = block_on(drive(
        None,
        None,
        |handle, _| Ok(async move { invoke(None, handle, arguments(None), 1, None)?.await }),
        async |_| {
            actual_calls += 1;
            Err::<ControlFlow<(), RootedValueSequence>, _>(
                crate::build_runtime_error("original evaluator error")
                    .with_identifier("OriginalEvaluator:Failure")
                    .build(),
            )
        },
    ))
    .unwrap_err();
    assert_eq!(actual_calls, 1);
    assert_eq!(error.identifier(), Some("OriginalEvaluator:Failure"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Legacy);
    let mut attempted = 0;
    let recovered = block_on(drive(
        None,
        None,
        |handle, _| {
            Ok(async move {
                let error = invoke(None, handle.clone(), arguments(None), 1, None)?
                    .await
                    .unwrap_err();
                assert_eq!(error.identifier(), Some("OriginalEvaluator:Caught"));
                invoke(None, handle, arguments(None), 1, None)?.await
            })
        },
        async |_| {
            attempted += 1;
            if attempted == 1 {
                Err(crate::build_runtime_error("catchable original error")
                    .with_identifier("OriginalEvaluator:Caught")
                    .build())
            } else {
                Ok(ControlFlow::<(), _>::Continue(returned()?))
            }
        },
    ))
    .unwrap();
    assert!(matches!(recovered, ControlFlow::Continue(_)));
    assert_eq!(
        attempted, 2,
        "only the child's explicit second call is another evaluation"
    );
    let mut selected = 0;
    let stopped = block_on(drive(
        None,
        None,
        |handle, _| {
            Ok(async move {
                let first = invoke(None, handle.clone(), arguments(None), 1, None)?;
                let second = invoke(None, handle, arguments(None), 1, None)?;
                let (first, second) = futures::join!(first, second);
                first?;
                second
            })
        },
        async |_| {
            selected += 1;
            Ok(ControlFlow::Break(17usize))
        },
    ))
    .unwrap();
    assert!(matches!(stopped, ControlFlow::Break(17)));
    assert_eq!(
        selected, 1,
        "a domain budget stop cannot dispatch the queued second evaluation"
    );
}

struct DropObserved {
    live: ScopedFunctionHandle,
    dropped: Rc<Cell<bool>>,
}
struct ReadyDrop(DropObserved);
impl Future for ReadyDrop {
    type Output = BuiltinResult<RootedValueSequence>;
    fn poll(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        // The retained child field is intentional: its destructor observes
        // revocation even when the poll scope eagerly retires a Ready future.
        let _retained = &self.0;
        Poll::Ready(returned())
    }
}
#[test]
fn ready_child_is_revoked_before_original_poll_scope_destroys_it() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let token = runtime.cancellation();
    for selected in [false, true] {
        let dropped = Rc::new(Cell::new(false));
        let observed = dropped.clone();
        let result = block_on(drive(
            selected.then_some(&runtime),
            selected.then_some(&token),
            |handle, _| {
                Ok(ReadyDrop(DropObserved {
                    live: handle,
                    dropped: observed,
                }))
            },
            async |_| -> BuiltinResult<ControlFlow<(), RootedValueSequence>> {
                panic!("no nested request")
            },
        ))
        .unwrap();
        assert!(matches!(result, ControlFlow::Continue(_)));
        assert!(dropped.get());
    }
}
impl Future for DropObserved {
    type Output = BuiltinResult<RootedValueSequence>;
    fn poll(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}
impl Drop for DropObserved {
    fn drop(&mut self) {
        assert!(
            !self.live.is_live(),
            "scope revocation precedes child destructor"
        );
        self.dropped.set(true);
    }
}

#[test]
fn cancellation_and_unpolled_owner_drop_retire_original_child_and_capability_once() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    let source = CancellationSource::new();
    let runtime = crate::user_functions::test_support::exact_fixture_runtime()
        .with_cancellation_token(source.token());
    let token = source.token();
    let dropped = Rc::new(Cell::new(false));
    let saved = Rc::new(RefCell::new(None));
    let issued = saved.clone();
    let observed = dropped.clone();
    let mut call = Box::pin(drive(
        Some(&runtime),
        Some(&token),
        |handle, _| {
            *issued.borrow_mut() = Some(handle.clone());
            Ok(DropObserved {
                live: handle,
                dropped: observed,
            })
        },
        async |_| Ok(ControlFlow::<(), _>::Continue(returned()?)),
    ));
    assert!(matches!(
        call.as_mut()
            .poll(&mut Context::from_waker(noop_waker_ref())),
        Poll::Pending
    ));
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = call
        .as_mut()
        .poll(&mut Context::from_waker(noop_waker_ref()))
    else {
        panic!("original cancellation")
    };
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(dropped.get());
    assert!(!saved.borrow().as_ref().unwrap().is_live());
    let constructed = Cell::new(false);
    let error = block_on(drive::<_, ()>(
        Some(&runtime),
        Some(&token),
        |_, _| {
            constructed.set(true);
            Ok(async { returned() })
        },
        async |_| Ok(ControlFlow::Continue(returned()?)),
    ))
    .unwrap_err();
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(
        !constructed.get(),
        "pre-cancelled original scope must not construct its child"
    );
    let invoked = Cell::new(false);
    let unpolled = drive::<_, ()>(
        None,
        None,
        |_, _| {
            invoked.set(true);
            Ok(async { returned() })
        },
        async |_| Ok(ControlFlow::Continue(returned()?)),
    );
    drop(unpolled);
    assert!(
        !invoked.get(),
        "dropping an unpolled driver never issues a scope or child"
    );
}

#[test]
fn ordinary_feval_and_repeatable_selection_preserve_capability_and_refuse_transport() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let token = runtime.cancellation();
    let mut calls = 0;
    let result = block_on(drive(
        Some(&runtime),
        Some(&token),
        |handle, original| {
            Ok(async move {
                let value = Value::ScopedFunctionHandle(handle.clone());
                assert!(value.is_function_handle());
                assert_eq!(value.clone(), value);
                assert!(crate::foreign::reject_scoped_callback(&value).is_err());
                assert!(crate::execution::value_codec::encode_inline_value(&value).is_err());
                let display =
                    crate::builtins::introspection::function_handle_text::dispatch_func2str(
                        value.clone(),
                    )?;
                let named =
                    crate::builtins::introspection::function_handle_text::dispatch_str2func(
                        display,
                    )?;
                assert_ne!(named, value, "text never reconstructs capability authority");
                let info = crate::builtins::introspection::function_metadata::dispatch_functions(
                    value.clone(),
                )?;
                let Value::Struct(info) = info else {
                    panic!("functions metadata")
                };
                assert_eq!(
                    info.fields.get("type"),
                    Some(&Value::string("scopedcallable"))
                );
                let original = original.as_ref().unwrap();
                // Actual canonical feval selects and retains the typed target,
                // then the original owner pumps its one nested evaluator request.
                crate::call::feval::invoke_retained(
                    Some(original),
                    value.clone(),
                    RootedValues::in_context(vec![Value::Num(3.0)], original)?.into(),
                    1,
                    None,
                )
                .await?;
                // Optional resolution must execute the same issued target,
                // not decline it or fall back to textual name resolution.
                let result = crate::call::descriptor::try_execute_callable_descriptor_in_context(
                    original,
                    crate::call::descriptor::CallableDescriptor {
                        target: crate::call::descriptor::CallableTarget::Scoped(handle),
                        args: vec![Value::Num(3.0)],
                        requested_outputs: 1,
                        metadata: crate::call::descriptor::CallableMetadata::default(),
                    },
                )
                .await?;
                Ok(result.expect("the scoped target was selected"))
            })
        },
        async |request| {
            calls += 1;
            assert_eq!(request.arguments.view().get(0), Some(&Value::Num(3.0)));
            Ok(ControlFlow::<(), _>::Continue(returned()?))
        },
    ))
    .unwrap();
    assert!(matches!(result, ControlFlow::Continue(_)));
    assert_eq!(calls, 2);
}
