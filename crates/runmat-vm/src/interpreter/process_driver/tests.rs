use super::*;
use futures::task::noop_waker;
use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
use runmat_runtime::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use runmat_runtime::execution::ExecutionServiceError;
use runmat_value::{HandleRef, Value};
use std::cell::RefCell;
use std::error::Error;

#[test]
fn selected_reentry_reuses_exact_poll_owner_and_runs_as_a_sibling() {
    for managed in [false, true] {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let mut runtime = RuntimeContext::with_cancellation(
            Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        );
        if managed {
            runtime =
                runtime.with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(
                    ManagedParallel(ManagedHostEffectLedger::new(8 * 1024 * 1024)),
                )));
        }
        assert!(ProcessDriverHandle::current_in_context(&runtime)
            .unwrap()
            .is_none());
        let owner =
            ProcessContinuationDriver::new_with_dynamic_admission_in_context(&runtime, None)
                .unwrap();
        let current = runtime.clone();
        let unrelated = RuntimeContext::with_cancellation(
            Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        );
        let changed_generation = runtime.clone().with_cancellation_token(
            runmat_execution::cancellation::CancellationSource::new().token(),
        );
        let root_poll = Rc::new(Cell::new(false));
        let active = root_poll.clone();
        let driven = owner.drive_in_context(&runtime, move || {
            let pending = async move {
                assert!(ProcessDriverHandle::current_in_context(&unrelated).is_err());
                assert!(ProcessDriverHandle::current_in_context(&changed_generation).is_err());
                let handle = ProcessDriverHandle::current_in_context(&current)?
                    .expect("original root poll owns reentry");
                let child_runtime = current.clone();
                let child_handle = handle.clone();
                let child_active = active.clone();
                handle
                    .enqueue_owned_in_context(&current, move || async move {
                        assert!(
                            !child_active.get(),
                            "callee must not retain its caller's poll stack"
                        );
                        let selected = ProcessDriverHandle::current_in_context(&child_runtime)?
                            .expect("original child poll owns reentry");
                        let selected_owner = selected.owner()?;
                        let original_owner = child_handle.owner()?;
                        assert!(
                            match (&selected_owner, &original_owner) {
                                (
                                    DriverStateOwner::Legacy(left),
                                    DriverStateOwner::Legacy(right),
                                ) => Rc::ptr_eq(left, right),
                                (
                                    DriverStateOwner::Managed(left),
                                    DriverStateOwner::Managed(right),
                                ) => left.ptr_eq(right),
                                _ => false,
                            },
                            "selected child must reuse the original physical driver"
                        );
                        assert_eq!(
                            selected.owner()?.borrow().current,
                            child_handle.owner()?.borrow().current
                        );
                        assert!(selected.owner()?.borrow().current.is_some());
                        Ok::<_, RuntimeError>(42)
                    })?
                    .await
            };
            let mut pending = Box::pin(pending);
            futures::future::poll_fn(move |context| {
                root_poll.set(true);
                let result = pending.as_mut().poll(context);
                root_poll.set(false);
                result
            })
        });
        let driven = driven.unwrap();
        assert_eq!(futures::executor::block_on(driven).unwrap(), 42);
        assert!(ProcessDriverHandle::current_in_context(&runtime)
            .unwrap()
            .is_none());
    }
}

#[test]
fn selected_driver_handoff_restores_after_pending_and_panic() {
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        runmat_execution::cancellation::CancellationSource::new().token(),
    );
    let owner =
        ProcessContinuationDriver::new_with_dynamic_admission_in_context(&runtime, None).unwrap();
    let current = runtime.clone();
    let mut pending = Box::pin(
        owner
            .drive_in_context(&runtime, move || async move {
                assert!(ProcessDriverHandle::current_in_context(&current)
                    .unwrap()
                    .is_some());
                futures::future::pending::<()>().await
            })
            .unwrap(),
    );
    assert!(pending
        .as_mut()
        .poll(&mut Context::from_waker(&noop_waker()))
        .is_pending());
    assert!(ProcessDriverHandle::current_in_context(&runtime)
        .unwrap()
        .is_none());
    drop(pending);
    drop(owner);
    let independent =
        ProcessContinuationDriver::new_with_dynamic_admission_in_context(&runtime, None).unwrap();
    let current = runtime.clone();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        futures::executor::block_on(
            independent
                .drive_in_context(&runtime, move || async move {
                    assert!(ProcessDriverHandle::current_in_context(&current)
                        .unwrap()
                        .is_some());
                    panic!("selected root panic witness")
                })
                .unwrap(),
        )
    }));
    assert!(unwind.is_err());
    assert!(ProcessDriverHandle::current_in_context(&runtime)
        .unwrap()
        .is_none());
}

struct Retired(&'static str, Rc<RefCell<Vec<&'static str>>>);

struct CatchScopedProbe {
    runtime: RuntimeContext,
    catch: Option<Rc<runmat_value::MException>>,
    polls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
}

impl CatchScopedProbe {
    fn assert_selected_catch(&self) {
        assert_eq!(
            self.runtime
                .active_catch_exception_shared()
                .as_ref()
                .map(Rc::as_ptr),
            self.catch.as_ref().map(Rc::as_ptr),
            "the exact selected exception, including explicit None, survives sibling placement"
        );
    }
}

impl Future for CatchScopedProbe {
    type Output = SemanticReply;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.assert_selected_catch();
        self.polls.set(self.polls.get() + 1);
        Poll::Pending
    }
}

impl Drop for CatchScopedProbe {
    fn drop(&mut self) {
        self.assert_selected_catch();
        self.drops.set(self.drops.get() + 1);
    }
}

struct ScopedFutureDrop {
    scope_id: runmat_execution::ExecutionScopeId,
    dropped: Rc<Cell<bool>>,
}

impl Drop for ScopedFutureDrop {
    fn drop(&mut self) {
        assert_eq!(
            runmat_runtime::context::legacy::active()
                .expect("failed scoped future drops under its selected runtime")
                .execution()
                .scope_id(),
            self.scope_id
        );
        self.dropped.set(true);
    }
}

struct ManagedParallel(ManagedHostEffectLedger);

impl RuntimeParallelService for ManagedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

impl Drop for Retired {
    fn drop(&mut self) {
        self.1.borrow_mut().push(self.0);
    }
}

fn assert_runtime_cancellation(error: &RuntimeError) {
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source()
            .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
        Some(ExecutionServiceError::Cancelled)
    ));
}

#[test]
fn pending_child_retires_before_awaiting_root() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let root_retired = Rc::clone(&retired);
    let child_retired = Rc::clone(&retired);
    let mut driven = Box::pin(owner.drive(async move {
        let _root = Retired("root", root_retired);
        handle
            .enqueue(async move {
                let _child = Retired("child", child_retired);
                futures::future::pending::<()>().await;
                unreachable!("suspended child was cancelled")
            })
            .expect("live invocation")
            .await
    }));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(driven.as_mut().poll(&mut context), Poll::Pending));
    drop(driven);
    assert_eq!(*retired.borrow(), ["child", "root"]);

    // A semantic sibling must not depend on the caller's catch guard still
    // being active. Both an unpolled factory's captures and a pending child
    // retire under the exact selected Rc, or explicitly no catch.
    for has_catch in [false, true] {
        for unpolled in [false, true] {
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let selected = has_catch.then(|| {
                Rc::new(runmat_value::MException::new(
                    "RunMat:selected".into(),
                    "selected".into(),
                ))
            });
            // Unlike the legacy enqueue witness above, this selected scoped
            // queue requires the driver to own this runtime's authority.
            let owner =
                ProcessContinuationDriver::new_with_dynamic_admission_in_context(&runtime, None)
                    .expect("select the catch matrix's process driver runtime");
            let handle = owner.handle();
            let polls = Rc::new(Cell::new(0));
            let drops = Rc::new(Cell::new(0));
            let probe = CatchScopedProbe {
                runtime: runtime.clone(),
                catch: selected.clone(),
                polls: polls.clone(),
                drops: drops.clone(),
            };
            let child_runtime = runtime.clone();
            let mut driven = Box::pin(owner.drive(async move {
                let guard = child_runtime.enter_active_catch_shared(selected);
                let scope = child_runtime.prepare_poll_scope()?;
                let ticket =
                    handle.enqueue_scoped_in_context(&child_runtime, scope, move || probe)?;
                drop(guard);
                if unpolled {
                    drop(ticket);
                    futures::future::pending::<SemanticReply>().await
                } else {
                    ticket.await
                }
            }));
            let outer = Rc::new(runmat_value::MException::new(
                "RunMat:unrelated".into(),
                "unrelated".into(),
            ));
            let guard = runtime.enter_active_catch_shared(Some(outer.clone()));
            match driven.as_mut().poll(&mut context) {
                Poll::Pending => {}
                Poll::Ready(Err(error)) => panic!(
                    "catch matrix admission failed (has_catch={has_catch}, unpolled={unpolled}): {error:?}"
                ),
                Poll::Ready(Ok(_)) => panic!(
                    "catch matrix unexpectedly completed (has_catch={has_catch}, unpolled={unpolled})"
                ),
            }
            drop(driven);
            assert_eq!(polls.get(), usize::from(!unpolled));
            assert_eq!(drops.get(), 1);
            assert!(Rc::ptr_eq(
                &runtime
                    .active_catch_exception_shared()
                    .expect("restore outer"),
                &outer,
            ));
            drop(guard);
            assert!(runtime.active_catch_exception_shared().is_none());
        }
    }
}

#[test]
fn canceled_ancestor_retires_descendants_deepest_first() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let root_retired = Rc::clone(&retired);
    let child_retired = Rc::clone(&retired);
    let grandchild_retired = Rc::clone(&retired);
    let child_handle = handle.clone();
    let mut driven = Box::pin(owner.drive(async move {
        let _root = Retired("root", root_retired);
        let child = handle.enqueue(async move {
            let _child = Retired("child", child_retired);
            child_handle
                .enqueue(async move {
                    let _grandchild = Retired("grandchild", grandchild_retired);
                    futures::future::pending::<()>().await;
                    unreachable!("suspended grandchild was cancelled")
                })?
                .await
        })?;
        child.await
    }));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(driven.as_mut().poll(&mut context), Poll::Pending));
    drop(driven);
    assert_eq!(*retired.borrow(), ["grandchild", "child", "root"]);
}

#[test]
fn ready_root_retires_orphaned_child_before_root() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let root = Retired("root", Rc::clone(&retired));
    let mut child_retired = Some(Rc::clone(&retired));
    let mut driven = Box::pin(owner.drive(futures::future::poll_fn(move |_| {
        let _keep_root_until_driver_retirement = &root;
        let child = Retired("child", child_retired.take().expect("first root poll only"));
        let ticket = handle
            .enqueue(async move {
                let _child = child;
                futures::future::pending::<()>().await;
                unreachable!("orphaned child must retire")
            })
            .expect("root may enqueue before completion");
        drop(ticket);
        Poll::Ready(())
    })));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(
        driven.as_mut().poll(&mut context),
        Poll::Ready(())
    ));
    assert_eq!(*retired.borrow(), ["child", "root"]);
}

#[test]
fn ready_child_retires_orphaned_grandchild_before_reply() {
    for owned in [false, true] {
        let owner = ProcessContinuationDriver::new();
        let handle = owner.handle();
        let retired = Rc::new(RefCell::new(Vec::new()));
        let observed_retirement = Rc::clone(&retired);
        let child = Retired("child", Rc::clone(&retired));
        let mut grandchild_retired = Some(Rc::clone(&retired));
        let child_handle = handle.clone();
        let result = futures::executor::block_on(owner.drive(async move {
            let child = futures::future::poll_fn(move |_| {
                let grandchild = Retired(
                    "grandchild",
                    grandchild_retired.take().expect("first child poll only"),
                );
                let ticket = child_handle
                    .enqueue(async move {
                        let _grandchild = grandchild;
                        futures::future::pending::<()>().await;
                        unreachable!("orphaned grandchild must retire")
                    })
                    .expect("child may enqueue before completion");
                drop(ticket);
                let outputs = RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error());
                let captures =
                    RootedValues::new(Vec::new()).map_err(|error| error.into_runtime_error());
                let result =
                    outputs.and_then(|outputs| captures.map(|captures| (outputs, captures)));
                // Keep the child's owned state in its future until the driver retires it.
                let _ = &child;
                Poll::Ready(result)
            });
            let result = if owned {
                handle.enqueue_owned(child)?.await?
            } else {
                handle.enqueue(child)?.await?
            };
            assert_eq!(*observed_retirement.borrow(), ["grandchild", "child"]);
            Ok::<_, RuntimeError>(result)
        }))
        .expect("ready child returns one typed result");
        assert!(result.0.value().is_empty());
        assert_eq!(*retired.borrow(), ["grandchild", "child"]);
    }
}

#[test]
fn completed_child_transfers_one_typed_reply() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let result = futures::executor::block_on(owner.drive(async move {
        handle
            .enqueue(async {
                let outputs = RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())?;
                let captures =
                    RootedValues::new(Vec::new()).map_err(|error| error.into_runtime_error())?;
                Ok((outputs, captures))
            })?
            .await
    }))
    .expect("child completion");
    assert!(result.0.value().is_empty());
    assert!(result.1.values().is_empty());
}

#[test]
fn ready_sibling_advances_past_independent_pending_child() {
    let owner = ProcessContinuationDriver::new();
    let handle = owner.handle();
    let retired = Rc::new(RefCell::new(Vec::new()));
    let pending_retired = Rc::clone(&retired);
    let completed = futures::executor::block_on(owner.drive(async move {
        let pending = handle.enqueue(async move {
            let _pending = Retired("pending", pending_retired);
            futures::future::pending::<()>().await;
            unreachable!("independent pending child was not released")
        })?;
        let mut yielded = false;
        futures::future::poll_fn(|context| {
            if std::mem::replace(&mut yielded, true) {
                Poll::Ready(())
            } else {
                context.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await;
        let ready = handle.enqueue(async {
            Ok((
                RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())?,
                RootedValues::new(Vec::new()).map_err(|error| error.into_runtime_error())?,
            ))
        })?;
        let result = ready.await?;
        drop(pending);
        Ok::<_, RuntimeError>(result)
    }))
    .expect("ready sibling makes progress without the pending sibling");
    assert!(completed.0.value().is_empty());
    assert_eq!(*retired.borrow(), ["pending"]);
}

#[test]
fn queued_arguments_are_rooted_before_the_first_child_poll() {
    let initial = runmat_gc::gc_allocate_rooted(Value::String("queued input".into()))
        .expect("allocate managed input");
    let handle = initial.handle();
    let arguments = RootedValues::new(vec![Value::HandleObject(HandleRef {
        class_name: "QueuedInput".into(),
        target: handle,
        valid: true,
    })])
    .expect("register selected arguments before enqueue");
    initial
        .unroot()
        .expect("transfer input root to queued call");
    let owner = ProcessContinuationDriver::new();
    let ticket = owner
        .handle()
        .enqueue(async move {
            let _arguments = arguments;
            unreachable!("this witness does not poll the child")
        })
        .expect("queue owned call");
    runmat_gc::gc_collect_major().expect("collect before first child poll");
    assert_eq!(
        runmat_gc::gc_clone_value(&handle).expect("queued argument owns its member"),
        Value::String("queued input".into())
    );
    drop(ticket);
    drop(owner);
    runmat_gc::gc_collect_major().expect("collect after queued call retirement");
    assert!(runmat_gc::gc_clone_value(&handle).is_err());
}

#[test]
fn managed_queue_reuses_one_exact_task_slot_across_sequential_children() {
    let ledger = ManagedHostEffectLedger::new(8 * 1024 * 1024);
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let owner = ProcessContinuationDriver::new_with_dynamic_admission_in_context(&runtime, None)
        .expect("admit managed process driver");
    let handle = owner.handle();
    let child_runtime = runtime.clone();
    let child_cancellation = cancellation.clone();
    futures::executor::block_on(
        owner
            .drive_in_context(&runtime, move || async move {
                let pending = handle.enqueue_owned_in_context(&child_runtime, || async move {
                    futures::future::pending::<Result<usize, RuntimeError>>().await
                })?;
                for expected in 0..64_usize {
                    let actual = handle
                        .enqueue_owned_in_context(
                            &child_runtime,
                            move || async move { Ok(expected) },
                        )?
                        .await?;
                    assert_eq!(actual, expected);
                }
                let scope = child_runtime.prepare_poll_scope()?;
                let scoped = handle
                    .enqueue_owned_scoped_in_context(&child_runtime, scope, || async move {
                        Ok::<_, RuntimeError>(64_usize)
                    })?
                    .await?;
                assert_eq!(scoped, 64);
                assert!(
                    handle.owner()?.borrow().tasks.capacity() <= 2,
                    "one pending child plus sequential children reuse two exact slots"
                );
                let unpolled_constructed = Rc::new(Cell::new(false));
                let constructed_in_unpolled = Rc::clone(&unpolled_constructed);
                let unpolled_dropped = Rc::new(Cell::new(false));
                let dropped_in_unpolled = Rc::clone(&unpolled_dropped);
                let scope_id = child_runtime.execution().scope_id();
                let scope = child_runtime.prepare_poll_scope()?;
                let drop_probe = ScopedFutureDrop {
                    scope_id,
                    dropped: dropped_in_unpolled,
                };
                let unpolled =
                    handle.enqueue_owned_scoped_in_context(&child_runtime, scope, move || {
                        constructed_in_unpolled.set(true);
                        async move {
                            let _drop_probe = drop_probe;
                            Ok::<_, RuntimeError>(65_usize)
                        }
                    })?;
                assert!(!unpolled_constructed.get());
                drop(unpolled);
                // A later sibling makes the driver service the canceled entry;
                // its factory must be destroyed without ever running.
                let sibling = handle
                    .enqueue_owned_in_context(&child_runtime, || async move {
                        Ok::<_, RuntimeError>(66_usize)
                    })?
                    .await?;
                assert_eq!(sibling, 66);
                assert!(!unpolled_constructed.get());
                assert!(unpolled_dropped.get());
                let constructed = Rc::new(Cell::new(false));
                let constructed_in_factory = Rc::clone(&constructed);
                let dropped = Rc::new(Cell::new(false));
                let dropped_in_future = Rc::clone(&dropped);
                let scope_id = child_runtime.execution().scope_id();
                let scope = child_runtime.prepare_poll_scope()?;
                let refused =
                    handle.enqueue_owned_scoped_in_context(&child_runtime, scope, move || {
                        constructed_in_factory.set(true);
                        child_cancellation.cancel(runmat_execution::CancellationReason::User);
                        let drop_probe = ScopedFutureDrop {
                            scope_id,
                            dropped: dropped_in_future,
                        };
                        async move {
                            let _drop_probe = drop_probe;
                            Ok::<_, RuntimeError>(67_usize)
                        }
                    })?;
                assert!(!constructed.get(), "queued factory ran on the caller stack");
                let error = refused
                    .await
                    .expect_err("post-factory cancellation must remain typed");
                assert!(constructed.get());
                assert!(dropped.get());
                assert_runtime_cancellation(&error);
                assert_eq!(handle.owner()?.borrow().tasks.ids().count(), 1);
                drop(pending);
                Ok::<_, RuntimeError>(())
            })
            .expect("admit managed root"),
    )
    .expect("all sequential children complete");
    // The cancellation probe above temporarily retains the pending child,
    // canceled unpolled entry, and its sibling concurrently. Geometric growth
    // admits four slots for that three-entry peak, not one slot per child.
    assert!(owner.owner.state().borrow().tasks.capacity() <= 4);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(owner);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    // A root that loses its authority after construction never reaches the
    // ready queue, and its captures retire under the context that admitted it.
    let refused_ledger = ManagedHostEffectLedger::new(8 * 1024 * 1024);
    let refused_cancellation = runmat_execution::cancellation::CancellationSource::new();
    let refused_runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        refused_cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedParallel(refused_ledger.clone()))),
    );
    let refused_owner =
        ProcessContinuationDriver::new_with_dynamic_admission_in_context(&refused_runtime, None)
            .expect("admit canceled-root driver");
    let constructed = Rc::new(Cell::new(false));
    let constructed_in_factory = Rc::clone(&constructed);
    let dropped = Rc::new(Cell::new(false));
    let dropped_in_future = Rc::clone(&dropped);
    let scope_id = refused_runtime.execution().scope_id();
    let error = match refused_owner.drive_in_context(&refused_runtime, move || {
        constructed_in_factory.set(true);
        refused_cancellation.cancel(runmat_execution::CancellationReason::User);
        let drop_probe = ScopedFutureDrop {
            scope_id,
            dropped: dropped_in_future,
        };
        async move {
            let _drop_probe = drop_probe;
            Ok::<_, RuntimeError>(())
        }
    }) {
        Ok(_) => panic!("canceled root must not be published"),
        Err(error) => error,
    };
    assert!(constructed.get());
    assert!(dropped.get());
    assert_runtime_cancellation(&error);
    assert_eq!(refused_owner.owner.state().borrow().tasks.ids().count(), 0);
    drop(refused_owner);
    assert_eq!(refused_ledger.snapshot().live_claimed_bytes, 0);
}
