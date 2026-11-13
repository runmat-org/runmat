use super::*;

#[cfg(feature = "plot-core")]
#[path = "tests/axes_mutations.rs"]
mod axes_mutations;
use crate::execution::RootedValueSequence;
use crate::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionService, RuntimeExecutionServices,
};
use futures::channel::oneshot;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_value::Value;
use std::cell::RefCell;
use std::future::Future;
use std::num::NonZeroU64;

struct PendingFirstTurn {
    receiver: RefCell<Option<oneshot::Receiver<()>>>,
    reached: Rc<Cell<usize>>,
    dropped: Rc<Cell<usize>>,
}

impl RuntimeExecutionServices for PendingFirstTurn {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap())
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        let Some(receiver) = self.receiver.borrow_mut().take() else {
            return Box::pin(async { Ok(()) });
        };
        self.reached.set(self.reached.get() + 1);
        let dropped = self.dropped.clone();
        struct TurnDrop(Rc<Cell<usize>>);
        impl Drop for TurnDrop {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        Box::pin(async move {
            let _witness = TurnDrop(dropped);
            receiver
                .await
                .map_err(|_| ExecutionServiceError::Infrastructure("test turn retired".into()))?;
            Ok(())
        })
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"graphics-callback-pending-turn"])
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        Ok(None)
    }
    fn ensure_pool(&self, _: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn close_pool(&self, _: &PoolHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn create_future(&self, _: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn spawn(&self, _: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_future(
        &self,
        _: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_task(
        &self,
        _: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn claim_next_task_result(
        &self,
        _: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn mark_task_result_read(&self, _: &TaskHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn begin_await(&self, _: Value) -> Result<AwaitAction, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn complete_future(
        &self,
        _: crate::execution::FutureCompletionRef<'_>,
        _: Result<RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn cancel(&self, _: &Value, _: CancellationReason) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn drain_scope(&self, _: CancellationReason) {}
}

fn selected() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn queue(capacity: usize) -> RuntimeGraphicsCallbackQueue {
    RuntimeGraphicsCallbackQueue::new(
        NonZeroUsize::new(capacity).expect("positive event capacity"),
        NonZeroUsize::new(4).expect("positive registration capacity"),
    )
}

#[cfg(feature = "plot-core")]
#[test]
fn production_registration_requires_live_receiver_and_selected_queue_owner() {
    use crate::builtins::plotting::state::{
        close_figure, figure_graphics_receiver, reset_plot_state, select_figure, FigureHandle,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let figure = FigureHandle::from(37);
    select_figure(figure);
    let receiver = figure_graphics_receiver(figure)
        .expect("issue receiver")
        .expect("live Figure");
    let base = selected();
    let queue_owner = RuntimeGraphicsCallbackQueue::new_for_selected(
        &base,
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(4).unwrap(),
    );
    let ports = base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(queue_owner.downgrade());
    let selected_context = base.with_service_ports(ports);
    let callback: Rc<GraphicsCallbackResponder> =
        Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Suppress(false)) }));
    let foreign_base = selected();
    let foreign_ports = foreign_base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(queue_owner.downgrade());
    let foreign = foreign_base.with_service_ports(foreign_ports);
    let Err(mismatch) = queue_owner.register_graphics_response(
        foreign,
        receiver.clone(),
        GraphicsCallbackKind::ZoomButtonDownFilter,
        callback.clone(),
    ) else {
        panic!("foreign session must not install a callable");
    };
    assert_eq!(
        mismatch.identifier(),
        Some("RunMat:graphics:CallbackSessionMismatch")
    );

    let forged = GraphicsReceiverRef::issued(receiver.kind(), receiver.generation());
    let Err(forged_error) = queue_owner.register_graphics_response(
        selected_context.clone(),
        forged,
        GraphicsCallbackKind::ZoomButtonDownFilter,
        callback.clone(),
    ) else {
        panic!("same numeric coordinates do not authenticate");
    };
    assert_eq!(
        forged_error.identifier(),
        Some("RunMat:graphics:CallbackSourceRetired")
    );

    let (registration, _ingress) = queue_owner
        .register_graphics_response(
            selected_context.clone(),
            receiver.clone(),
            GraphicsCallbackKind::ZoomButtonDownFilter,
            callback.clone(),
        )
        .expect("live same-session Figure");
    drop(registration);
    close_figure(Some(figure)).expect("close Figure");
    let Err(retired) = queue_owner.register_graphics_response(
        selected_context,
        receiver,
        GraphicsCallbackKind::ZoomButtonDownFilter,
        callback,
    ) else {
        panic!("closed Figure cannot re-register");
    };
    assert_eq!(
        retired.identifier(),
        Some("RunMat:graphics:CallbackSourceRetired")
    );
}

fn event(kind: GraphicsCallbackKind) -> GraphicsCallbackHostEvent {
    GraphicsCallbackHostEvent {
        figure: 7,
        kind,
        position: Some((2.0, 3.0)),
    }
}

#[cfg(feature = "plot-core")]
#[test]
fn bound_property_payload_pins_scene_before_and_after_callback_effects() {
    use crate::builtins::plotting::state::{
        axes_graphics_receiver_for_scene, capture_figure_persistence_snapshot,
        figure_graphics_receiver, reset_plot_state, select_figure, set_figure_name, FigureHandle,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    for (change_before, change_during) in [(false, false), (true, false), (false, true)] {
        reset_plot_state();
        let figure = FigureHandle::from(38);
        select_figure(figure);
        let receiver = figure_graphics_receiver(figure).unwrap().unwrap();
        let snapshot = capture_figure_persistence_snapshot(&[figure], 1, usize::MAX).unwrap();
        let scene = snapshot.entries[0].scene_lease.clone();
        let axes = axes_graphics_receiver_for_scene(&receiver, &scene, 0)
            .unwrap()
            .unwrap();
        let base = selected();
        let queue = RuntimeGraphicsCallbackQueue::new_for_selected(
            &base,
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(1).unwrap(),
        );
        let ports = base
            .service_ports()
            .clone()
            .with_graphics_callback_queue(queue.downgrade());
        let selected = base.with_service_ports(ports);
        let calls = Rc::new(Cell::new(0));
        let callback: Rc<GraphicsBoundCallbackResponder> = {
            let calls = calls.clone();
            Rc::new(move |_, _| {
                calls.set(calls.get() + 1);
                Box::pin(async move {
                    if change_during {
                        set_figure_name(figure, "callback changed scene".into()).unwrap();
                        Err(build_runtime_error("original callback failure")
                            .with_identifier("RunMat:test:OriginalCallbackFailure")
                            .build())
                    } else {
                        Ok(GraphicsCallbackResult::Completed)
                    }
                })
            })
        };
        let kind = GraphicsCallbackKind::PanActionPre;
        let ingress = queue
            .replace_graphics_bound_property_response(selected, receiver.clone(), kind, callback)
            .unwrap();
        let event = GraphicsCallbackHostEvent {
            figure: figure.as_u32(),
            kind,
            position: None,
        };
        assert!(
            ingress.request(event).is_err(),
            "metadata alone cannot authorize a bound role"
        );
        let ticket = ingress
            .request_bound(
                event,
                GraphicsCallbackRolePayload::Action {
                    figure: receiver,
                    scene,
                    axes,
                },
            )
            .unwrap();
        if change_before {
            set_figure_name(figure, "scene changed before dispatch".into()).unwrap();
        }
        let processed = futures::executor::block_on(queue.process());
        if change_during {
            assert_eq!(
                processed.unwrap_err().identifier(),
                Some("RunMat:test:OriginalCallbackFailure")
            );
            assert_eq!(
                ticket.disposition(),
                GraphicsCallbackDisposition::Indeterminate
            );
            assert_eq!(
                futures::executor::block_on(ticket),
                Err(GraphicsCallbackHostFailure::Indeterminate)
            );
            assert_eq!(ingress.revoke(), GraphicsCallbackDisposition::Indeterminate);
            assert!(futures::executor::block_on(queue.process()).is_err());
        } else {
            processed.unwrap();
            assert_eq!(
                futures::executor::block_on(ticket),
                if change_before {
                    Err(GraphicsCallbackHostFailure::Retired)
                } else {
                    Ok(GraphicsCallbackResult::Completed)
                }
            );
        }
        assert_eq!(calls.get(), usize::from(!change_before));
    }
}

#[cfg(feature = "plot-core")]
#[test]
fn property_replacement_fences_queued_old_claim_and_reuses_bounded_slot() {
    use crate::builtins::plotting::state::{
        figure_graphics_receiver, reset_plot_state, select_figure, FigureHandle,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let figure = FigureHandle::from(39);
    select_figure(figure);
    let receiver = figure_graphics_receiver(figure)
        .expect("issue receiver")
        .expect("selected Figure");
    let base = selected();
    let owner = RuntimeGraphicsCallbackQueue::new_for_selected(
        &base,
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );
    let ports = base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(owner.downgrade());
    let selected = base.with_service_ports(ports);
    let old_calls = Rc::new(Cell::new(0));
    let old_callback: Rc<GraphicsCallbackResponder> = {
        let old_calls = old_calls.clone();
        Rc::new(move |_| {
            old_calls.set(old_calls.get() + 1);
            Box::pin(async { Ok(GraphicsCallbackResult::Suppress(false)) })
        })
    };
    let kind = GraphicsCallbackKind::ZoomButtonDownFilter;
    let old_ingress = owner
        .replace_graphics_property_response(selected.clone(), receiver.clone(), kind, old_callback)
        .expect("first property owner");
    let old_ticket = old_ingress
        .request(GraphicsCallbackHostEvent {
            figure: figure.as_u32(),
            kind,
            position: None,
        })
        .expect("queue old gesture");
    let new_calls = Rc::new(Cell::new(0));
    let new_callback: Rc<GraphicsCallbackResponder> = {
        let new_calls = new_calls.clone();
        Rc::new(move |_| {
            new_calls.set(new_calls.get() + 1);
            Box::pin(async { Ok(GraphicsCallbackResult::Suppress(true)) })
        })
    };
    let new_ingress = owner
        .replace_graphics_property_response(selected, receiver.clone(), kind, new_callback)
        .expect("replacement reuses the one admitted slot");
    assert!(!old_ingress.is_live());
    assert_eq!(owner.0.registrations.borrow().len(), 1);
    let new_ticket = new_ingress
        .request(GraphicsCallbackHostEvent {
            figure: figure.as_u32(),
            kind,
            position: None,
        })
        .expect("queue new gesture");
    futures::executor::block_on(owner.process()).expect("drain both envelopes");
    assert_eq!(
        futures::executor::block_on(old_ticket),
        Err(GraphicsCallbackHostFailure::Retired)
    );
    assert_eq!(
        futures::executor::block_on(new_ticket),
        Ok(GraphicsCallbackResult::Suppress(true))
    );
    assert_eq!(old_calls.get(), 0);
    assert_eq!(new_calls.get(), 1);
    assert!(owner.clear_graphics_property_response(&receiver, kind));
    assert!(!new_ingress.is_live());
    assert!(owner.0.registrations.borrow().is_empty());
}

#[cfg(feature = "plot-core")]
#[test]
fn replaced_callback_destructor_can_retire_another_property() {
    use crate::builtins::plotting::state::{
        figure_graphics_receiver, reset_plot_state, select_figure, FigureHandle,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    struct RetireOnDrop {
        owner: WeakRuntimeGraphicsCallbackQueue,
        receiver: GraphicsReceiverRef,
        retired: Rc<Cell<bool>>,
    }
    impl Drop for RetireOnDrop {
        fn drop(&mut self) {
            self.retired.set(self.owner.upgrade().is_some_and(|owner| {
                owner.clear_graphics_property_response(
                    &self.receiver,
                    GraphicsCallbackKind::PanActionPost,
                )
            }));
        }
    }

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let figure = FigureHandle::from(41);
    select_figure(figure);
    let receiver = figure_graphics_receiver(figure)
        .expect("issue receiver")
        .expect("selected Figure");
    let base = selected();
    let owner = RuntimeGraphicsCallbackQueue::new_for_selected(
        &base,
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    );
    let ports = base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(owner.downgrade());
    let selected = base.with_service_ports(ports);
    let callback: Rc<GraphicsCallbackResponder> =
        Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Completed) }));
    let other = owner
        .replace_graphics_property_response(
            selected.clone(),
            receiver.clone(),
            GraphicsCallbackKind::PanActionPost,
            callback,
        )
        .expect("second property");
    let retired = Rc::new(Cell::new(false));
    let witness = RetireOnDrop {
        owner: owner.downgrade(),
        receiver: receiver.clone(),
        retired: retired.clone(),
    };
    let old: Rc<GraphicsCallbackResponder> = Rc::new(move |_| {
        let _retain_until_drop = &witness;
        Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
    });
    owner
        .replace_graphics_property_response(
            selected.clone(),
            receiver.clone(),
            GraphicsCallbackKind::PanActionPre,
            old,
        )
        .expect("first property");
    let replacement: Rc<GraphicsCallbackResponder> =
        Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Completed) }));
    let current = owner
        .replace_graphics_property_response(
            selected,
            receiver,
            GraphicsCallbackKind::PanActionPre,
            replacement,
        )
        .expect("replacement releases old owner outside queue borrows");
    assert!(retired.get(), "old callback destructor ran");
    assert!(
        !other.is_live(),
        "reentrant retirement selected only its key"
    );
    assert!(current.is_live());
    assert_eq!(owner.0.registrations.borrow().len(), 1);
}

#[cfg(feature = "plot-core")]
#[test]
fn direct_registration_drop_releases_map_before_callback_destructor() {
    use crate::builtins::plotting::state::{
        figure_graphics_receiver, reset_plot_state, select_figure, FigureHandle,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    struct ClearOnDrop {
        queue: WeakRuntimeGraphicsCallbackQueue,
        receiver: GraphicsReceiverRef,
        observed: Rc<Cell<bool>>,
    }
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            self.observed.set(self.queue.upgrade().is_some_and(|queue| {
                queue.clear_graphics_property_response(
                    &self.receiver,
                    GraphicsCallbackKind::ZoomActionPost,
                )
            }));
        }
    }

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let figure = FigureHandle::from(42);
    select_figure(figure);
    let receiver = figure_graphics_receiver(figure)
        .expect("issue receiver")
        .expect("live Figure");
    let base = selected();
    let queue = RuntimeGraphicsCallbackQueue::new_for_selected(
        &base,
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    );
    let ports = base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(queue.downgrade());
    let selected = base.with_service_ports(ports);
    let other: Rc<GraphicsCallbackResponder> =
        Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Completed) }));
    let other_ingress = queue
        .replace_graphics_property_response(
            selected.clone(),
            receiver.clone(),
            GraphicsCallbackKind::ZoomActionPost,
            other,
        )
        .expect("other property");
    let observed = Rc::new(Cell::new(false));
    let destructor = ClearOnDrop {
        queue: queue.downgrade(),
        receiver: receiver.clone(),
        observed: observed.clone(),
    };
    let callback: Rc<GraphicsCallbackResponder> = Rc::new(move |_| {
        let _retain_until_drop = &destructor;
        Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
    });
    let (direct, _) = queue
        .register_graphics_response(
            selected,
            receiver,
            GraphicsCallbackKind::ZoomActionPre,
            callback,
        )
        .expect("direct owner");
    drop(direct);
    assert!(observed.get(), "callback destructor reentered the queue");
    assert!(!other_ingress.is_live());
    assert!(queue.0.registrations.borrow().is_empty());
}

#[cfg(feature = "plot-core")]
#[test]
fn retired_figure_property_roots_prune_before_numeric_reuse() {
    use crate::builtins::plotting::state::{
        close_figure, figure_graphics_receiver, reset_plot_state, select_figure, FigureHandle,
    };
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_plot_state();
    let figure = FigureHandle::from(40);
    select_figure(figure);
    let prior = figure_graphics_receiver(figure)
        .expect("issue receiver")
        .expect("selected Figure");
    let base = selected();
    let owner = RuntimeGraphicsCallbackQueue::new_for_selected(
        &base,
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );
    let ports = base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(owner.downgrade());
    let selected = base.with_service_ports(ports);
    let callback: Rc<GraphicsCallbackResponder> =
        Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Suppress(false)) }));
    let kind = GraphicsCallbackKind::PanButtonDownFilter;
    let old = owner
        .replace_graphics_property_response(selected.clone(), prior.clone(), kind, callback.clone())
        .expect("old exact property");
    close_figure(Some(figure)).expect("retire original Figure");
    select_figure(figure);
    let replacement = figure_graphics_receiver(figure)
        .expect("issue replacement")
        .expect("reused numeric Figure");
    assert!(!prior.shares_lease(&replacement));
    let new = owner
        .replace_graphics_property_response(selected, replacement.clone(), kind, callback)
        .expect("retired slot is pruned before new admission");
    assert!(!old.is_live());
    assert!(new.is_live());
    assert!(!owner.clear_graphics_property_response(&prior, kind));
    assert!(owner.clear_graphics_property_response(&replacement, kind));
    assert!(!new.is_live());
}

#[test]
fn deferred_events_keep_order_origin_and_reentrant_arrivals() {
    let queue = queue(2);
    let runtime = selected();
    let owner = runtime.local_identity();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let (release, gate) = oneshot::channel::<()>();
    let gate = Rc::new(RefCell::new(Some(gate)));
    let callback: Rc<GraphicsCallbackHandler> = {
        let observed = observed.clone();
        Rc::new(move |host_event| {
            let active = crate::context::legacy::active().expect("scope before handler invocation");
            assert_eq!(active.local_identity(), owner);
            let observed = observed.clone();
            let gate = gate.borrow_mut().take();
            Box::pin(async move {
                let active = crate::context::legacy::active().expect("origin scope");
                assert_eq!(active.local_identity(), owner);
                observed.borrow_mut().push(host_event.position.unwrap().0);
                if let Some(gate) = gate {
                    gate.await.expect("release first callback");
                }
                Ok(())
            })
        })
    };
    let (_registration, ingress) = queue
        .register(runtime, 7, GraphicsCallbackKind::PanActionPre, callback)
        .expect("register selected callback");
    ingress
        .send(event(GraphicsCallbackKind::PanActionPre))
        .expect("first event");
    queue.defer();
    assert!(
        observed.borrow().is_empty(),
        "Defer must not invoke callback"
    );
    let mut processing = Box::pin(queue.process());
    futures::executor::block_on(futures::future::poll_fn(|context| {
        assert!(processing.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    }));
    ingress
        .send(GraphicsCallbackHostEvent {
            position: Some((4.0, 5.0)),
            ..event(GraphicsCallbackKind::PanActionPre)
        })
        .expect("reentrant arrival");
    release.send(()).expect("resume first callback");
    futures::executor::block_on(processing).expect("drain both");
    assert_eq!(*observed.borrow(), vec![2.0, 4.0]);
}

#[test]
fn dropping_pending_callback_quarantines_exact_event_without_replay() {
    let queue = queue(2);
    let invoked = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(0));
    struct PendingDrop(Rc<Cell<usize>>);
    impl Drop for PendingDrop {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let callback: Rc<GraphicsCallbackHandler> = {
        let invoked = invoked.clone();
        let dropped = dropped.clone();
        Rc::new(move |_| {
            invoked.set(invoked.get() + 1);
            let witness = PendingDrop(dropped.clone());
            Box::pin(async move {
                let _witness = witness;
                std::future::pending::<()>().await;
                Ok(())
            })
        })
    };
    let (_registration, ingress) = queue
        .register(selected(), 7, GraphicsCallbackKind::PanActionPre, callback)
        .unwrap();
    ingress
        .send(event(GraphicsCallbackKind::PanActionPre))
        .unwrap();
    ingress
        .send(GraphicsCallbackHostEvent {
            position: Some((9.0, 10.0)),
            ..event(GraphicsCallbackKind::PanActionPre)
        })
        .unwrap();
    let mut processing = Box::pin(queue.process());
    futures::executor::block_on(futures::future::poll_fn(|context| {
        assert!(processing.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    }));
    assert_eq!(invoked.get(), 1);
    drop(processing);
    assert_eq!(dropped.get(), 1, "pending callback future retires");
    let error = futures::executor::block_on(queue.process())
        .expect_err("uncertain callback cannot replay or skip to later work");
    assert_eq!(
        error.identifier(),
        Some("RunMat:graphics:CallbackIndeterminate")
    );
    assert!(error
        .message()
        .contains("registration 1, figure 7, kind PanActionPre"));
    assert_eq!(invoked.get(), 1);
    assert!(matches!(
        *queue.0.in_flight.borrow(),
        Some(InFlight::Indeterminate(ref envelope)) if envelope.event.position == Some((2.0, 3.0))
    ));
}

#[test]
fn preinvocation_head_retries_before_later_ingress() {
    let queue = queue(2);
    let seen = Rc::new(RefCell::new(Vec::new()));
    let reached = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(0));
    let (_release, gate) = oneshot::channel();
    let runtime = RuntimeContext::new(Rc::new(PendingFirstTurn {
        receiver: RefCell::new(Some(gate)),
        reached: reached.clone(),
        dropped: dropped.clone(),
    }));
    let callback: Rc<GraphicsCallbackHandler> = {
        let seen = seen.clone();
        Rc::new(move |event| {
            seen.borrow_mut().push(event.position.unwrap().0);
            Box::pin(async { Ok(()) })
        })
    };
    let (_registration, ingress) = queue
        .register(runtime, 7, GraphicsCallbackKind::ZoomActionPost, callback)
        .unwrap();
    ingress
        .send(event(GraphicsCallbackKind::ZoomActionPost))
        .unwrap();
    ingress
        .send(GraphicsCallbackHostEvent {
            position: Some((4.0, 5.0)),
            ..event(GraphicsCallbackKind::ZoomActionPost)
        })
        .unwrap();
    let mut processing = Box::pin(queue.process());
    futures::executor::block_on(futures::future::poll_fn(|context| {
        assert!(processing.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    }));
    assert_eq!(reached.get(), 1, "first event reached actual CPU turn");
    assert!(seen.borrow().is_empty(), "callback was not invoked");
    drop(processing);
    assert_eq!(dropped.get(), 1, "pending turn retired on drain drop");
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(*seen.borrow(), vec![2.0, 4.0]);
}

#[test]
fn wrong_event_identity_and_retired_registration_cannot_invoke() {
    let queue = queue(2);
    let calls = Rc::new(Cell::new(0));
    let callback: Rc<GraphicsCallbackHandler> = {
        let calls = calls.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            Box::pin(async { Ok(()) })
        })
    };
    let (registration, ingress) = queue
        .register(
            selected(),
            7,
            GraphicsCallbackKind::ZoomActionPost,
            callback,
        )
        .expect("register");
    assert!(ingress
        .send(event(GraphicsCallbackKind::PanActionPost))
        .is_err());
    ingress
        .send(event(GraphicsCallbackKind::ZoomActionPost))
        .expect("queued before retirement");
    drop(registration);
    futures::executor::block_on(queue.process()).expect("retired events ignored");
    assert_eq!(calls.get(), 0);
}

#[test]
fn dropping_owner_releases_selected_callback_without_gui_port_ownership() {
    struct DropWitness(Rc<Cell<usize>>);
    impl Drop for DropWitness {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let drops = Rc::new(Cell::new(0));
    let queue = queue(2);
    let witness = DropWitness(drops.clone());
    let callback: Rc<GraphicsCallbackHandler> = Rc::new(move |_| {
        let _ = &witness;
        Box::pin(async { Ok(()) })
    });
    let (registration, ingress) = queue
        .register(
            selected(),
            7,
            GraphicsCallbackKind::DataCursorUpdate,
            callback,
        )
        .expect("register");
    drop(queue);
    assert_eq!(drops.get(), 1, "queue retirement releases callback root");
    assert_eq!(
        ingress
            .send(event(GraphicsCallbackKind::DataCursorUpdate))
            .expect_err("retired owner")
            .failure,
        GraphicsCallbackIngressFailure::Closed
    );
    drop(registration);
    assert_eq!(drops.get(), 1);
}

#[test]
fn callback_failure_is_terminal_and_does_not_consume_later_event() {
    let queue = queue(2);
    let calls = Rc::new(Cell::new(0));
    let callback: Rc<GraphicsCallbackHandler> = {
        let calls = calls.clone();
        Rc::new(move |_| {
            let call = calls.get() + 1;
            calls.set(call);
            Box::pin(async move {
                if call == 1 {
                    Err(build_runtime_error("first callback failed")
                        .with_identifier("RunMat:graphics:CallbackFailure")
                        .build())
                } else {
                    Ok(())
                }
            })
        })
    };
    let (_registration, ingress) = queue
        .register(selected(), 7, GraphicsCallbackKind::ZoomActionPre, callback)
        .expect("register");
    ingress
        .send(event(GraphicsCallbackKind::ZoomActionPre))
        .unwrap();
    ingress
        .send(event(GraphicsCallbackKind::ZoomActionPre))
        .unwrap();
    let error = futures::executor::block_on(queue.process()).expect_err("first error");
    assert_eq!(error.identifier(), Some("RunMat:graphics:CallbackFailure"));
    assert_eq!(calls.get(), 1);
    futures::executor::block_on(queue.process()).expect("second remains queued");
    assert_eq!(calls.get(), 2);
}

#[test]
fn reentrant_drain_is_rejected_without_a_strong_queue_cycle() {
    let queue = queue(2);
    let weak = Rc::downgrade(&queue.0);
    let callback: Rc<GraphicsCallbackHandler> = Rc::new(move |_| {
        let weak = weak.clone();
        Box::pin(async move {
            let owner = RuntimeGraphicsCallbackQueue(weak.upgrade().expect("live owner"));
            let error = owner.process().await.expect_err("nested drain denied");
            assert_eq!(
                error.identifier(),
                Some("RunMat:graphics:CallbackDrainReentrant")
            );
            Ok(())
        })
    });
    let (_registration, ingress) = queue
        .register(selected(), 7, GraphicsCallbackKind::PanActionPost, callback)
        .expect("register");
    ingress
        .send(event(GraphicsCallbackKind::PanActionPost))
        .unwrap();
    futures::executor::block_on(queue.process()).expect("outer drain completes");
}

#[test]
fn finite_ingress_reports_backpressure_without_blocking_or_dropping_event() {
    let queue = queue(1);
    let calls = Rc::new(Cell::new(0));
    let callback: Rc<GraphicsCallbackHandler> = {
        let calls = calls.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            Box::pin(async { Ok(()) })
        })
    };
    let (_registration, ingress) = queue
        .register(
            selected(),
            7,
            GraphicsCallbackKind::DataCursorUpdate,
            callback,
        )
        .unwrap();
    let first = event(GraphicsCallbackKind::DataCursorUpdate);
    ingress.send(first).expect("one queued event");
    let second = GraphicsCallbackHostEvent {
        position: Some((5.0, 6.0)),
        ..first
    };
    let error = ingress.send(second).expect_err("finite queue is full");
    assert_eq!(error.event, second);
    assert_eq!(error.failure, GraphicsCallbackIngressFailure::Full);
    assert_eq!(calls.get(), 0, "backpressure cannot invoke on host lane");
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(calls.get(), 1);
    ingress.send(error.event).expect("host retries exact event");
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(calls.get(), 2);
}

#[test]
fn registration_capacity_fails_before_retaining_new_origin() {
    let queue = RuntimeGraphicsCallbackQueue::new(
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );
    let callback: Rc<GraphicsCallbackHandler> = Rc::new(|_| Box::pin(async { Ok(()) }));
    let (first, _) = queue
        .register(
            selected(),
            7,
            GraphicsCallbackKind::PanActionPre,
            callback.clone(),
        )
        .unwrap();
    let error = queue
        .register(
            selected(),
            8,
            GraphicsCallbackKind::PanActionPre,
            callback.clone(),
        )
        .err()
        .expect("second owner is not retained");
    assert_eq!(
        error.identifier(),
        Some("RunMat:graphics:CallbackRegistrationCapacity")
    );
    drop(first);
    queue
        .register(selected(), 8, GraphicsCallbackKind::PanActionPre, callback)
        .expect("retired slot is reusable without reusing identity");
}

#[test]
fn result_bearing_filter_returns_typed_suppression_without_replaying_gesture() {
    let queue = queue(2);
    let calls = Rc::new(Cell::new(0));
    let callback: Rc<GraphicsCallbackResponder> = {
        let calls = calls.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            Box::pin(async { Ok(GraphicsCallbackResult::Suppress(true)) })
        })
    };
    let (_registration, ingress) = queue
        .register_response(
            selected(),
            7,
            GraphicsCallbackKind::PanButtonDownFilter,
            callback,
        )
        .unwrap();
    let ticket = ingress
        .request(event(GraphicsCallbackKind::PanButtonDownFilter))
        .unwrap();
    assert_eq!(
        ticket.disposition(),
        GraphicsCallbackDisposition::NotPublished
    );
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(ticket.disposition(), GraphicsCallbackDisposition::Published);
    assert_eq!(
        futures::executor::block_on(ticket).unwrap(),
        GraphicsCallbackResult::Suppress(true)
    );
    assert_eq!(calls.get(), 1);
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(calls.get(), 1, "a terminal request cannot replay");
    assert_eq!(ingress.revoke(), GraphicsCallbackDisposition::Published);
}

#[test]
fn abandoned_queued_host_request_cannot_start_callback() {
    let queue = queue(2);
    let calls = Rc::new(Cell::new(0));
    let callback: Rc<GraphicsCallbackResponder> = {
        let calls = calls.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
        })
    };
    let (_registration, ingress) = queue
        .register_response(selected(), 7, GraphicsCallbackKind::PanActionPre, callback)
        .unwrap();
    let mut ticket = ingress
        .request(event(GraphicsCallbackKind::PanActionPre))
        .unwrap();
    assert_eq!(ticket.abandon(), GraphicsCallbackDisposition::NotPublished);
    drop(ticket);
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(calls.get(), 0);
}

#[test]
fn started_host_request_drop_is_indeterminate_and_never_replayed() {
    let queue = queue(2);
    let calls = Rc::new(Cell::new(0));
    let (release, gate) = oneshot::channel::<()>();
    let gate = Rc::new(RefCell::new(Some(gate)));
    let callback: Rc<GraphicsCallbackResponder> = {
        let calls = calls.clone();
        let gate = gate.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            let gate = gate.borrow_mut().take().expect("one callback invocation");
            Box::pin(async move {
                gate.await.expect("release callback");
                Ok(GraphicsCallbackResult::Completed)
            })
        })
    };
    let (_registration, ingress) = queue
        .register_response(selected(), 7, GraphicsCallbackKind::PanActionPre, callback)
        .unwrap();
    let mut ticket = ingress
        .request(event(GraphicsCallbackKind::PanActionPre))
        .unwrap();
    let mut processing = Box::pin(queue.process());
    futures::executor::block_on(futures::future::poll_fn(|context| {
        assert!(processing.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    }));
    assert_eq!(calls.get(), 1);
    assert_eq!(ticket.abandon(), GraphicsCallbackDisposition::Indeterminate);
    drop(ticket);
    release.send(()).unwrap();
    futures::executor::block_on(processing).unwrap();
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(calls.get(), 1);
}

#[test]
fn invalid_result_and_runtime_failure_remain_distinct_host_and_session_errors() {
    let queue = queue(2);
    let wrong: Rc<GraphicsCallbackResponder> =
        Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Completed) }));
    let (_registration, ingress) = queue
        .register_response(
            selected(),
            7,
            GraphicsCallbackKind::ZoomButtonDownFilter,
            wrong,
        )
        .unwrap();
    let ticket = ingress
        .request(event(GraphicsCallbackKind::ZoomButtonDownFilter))
        .unwrap();
    let error = futures::executor::block_on(queue.process()).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:graphics:CallbackResult"));
    assert_eq!(
        futures::executor::block_on(ticket),
        Err(GraphicsCallbackHostFailure::InvalidResult)
    );

    let failing: Rc<GraphicsCallbackResponder> = Rc::new(|_| {
        Box::pin(async {
            Err(build_runtime_error("original callback failure")
                .with_identifier("RunMat:graphics:OriginalFailure")
                .build())
        })
    });
    let (_registration, ingress) = queue
        .register_response(selected(), 7, GraphicsCallbackKind::PanActionPost, failing)
        .unwrap();
    let ticket = ingress
        .request(event(GraphicsCallbackKind::PanActionPost))
        .unwrap();
    let error = futures::executor::block_on(queue.process()).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:graphics:OriginalFailure"));
    assert_eq!(
        futures::executor::block_on(ticket),
        Err(GraphicsCallbackHostFailure::CallbackFailed)
    );
}

#[test]
fn replacement_during_pending_cooperation_excludes_callback_start() {
    let (release, gate) = oneshot::channel();
    let reached = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(0));
    let service = Rc::new(PendingFirstTurn {
        receiver: RefCell::new(Some(gate)),
        reached: reached.clone(),
        dropped: dropped.clone(),
    });
    let selected = RuntimeContext::new(service);
    let queue = queue(2);
    let calls = Rc::new(Cell::new(0));
    let callback: Rc<GraphicsCallbackResponder> = {
        let calls = calls.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
        })
    };
    let (_registration, ingress) = queue
        .register_response(selected, 7, GraphicsCallbackKind::PanActionPre, callback)
        .unwrap();
    let ticket = ingress
        .request(event(GraphicsCallbackKind::PanActionPre))
        .unwrap();
    let mut processing = Box::pin(queue.process());
    futures::executor::block_on(futures::future::poll_fn(|context| {
        assert!(processing.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    }));
    assert_eq!(reached.get(), 1);
    assert_eq!(calls.get(), 0);
    assert_eq!(ingress.revoke(), GraphicsCallbackDisposition::NotPublished);
    release.send(()).unwrap();
    futures::executor::block_on(processing).unwrap();
    assert_eq!(dropped.get(), 1);
    assert_eq!(calls.get(), 0);
    assert_eq!(
        ticket.disposition(),
        GraphicsCallbackDisposition::NotPublished
    );
    assert_eq!(
        futures::executor::block_on(ticket),
        Err(GraphicsCallbackHostFailure::Retired)
    );
    assert_eq!(
        ingress.send(event(GraphicsCallbackKind::PanActionPre)),
        Err(GraphicsCallbackIngressError {
            event: event(GraphicsCallbackKind::PanActionPre),
            failure: GraphicsCallbackIngressFailure::Retired,
        })
    );
}

#[test]
fn session_close_during_pending_cooperation_retires_ready_head_without_callback() {
    let (release, gate) = oneshot::channel();
    let reached = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(0));
    let service = Rc::new(PendingFirstTurn {
        receiver: RefCell::new(Some(gate)),
        reached: reached.clone(),
        dropped: dropped.clone(),
    });
    let selected = RuntimeContext::new(service);
    let owner = queue(2);
    let external = owner.clone();
    let calls = Rc::new(Cell::new(0));
    let callback: Rc<GraphicsCallbackResponder> = {
        let calls = calls.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
        })
    };
    let (_registration, ingress) = owner
        .register_response(selected, 7, GraphicsCallbackKind::PanActionPre, callback)
        .unwrap();
    let ticket = ingress
        .request(event(GraphicsCallbackKind::PanActionPre))
        .unwrap();
    let mut processing = Box::pin(owner.process());
    futures::executor::block_on(futures::future::poll_fn(|context| {
        assert!(processing.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    }));
    assert_eq!(reached.get(), 1);
    assert_eq!(calls.get(), 0);
    owner.close();
    assert!(external.is_closed());
    assert_eq!(
        ticket.disposition(),
        GraphicsCallbackDisposition::NotPublished
    );
    assert_eq!(
        futures::executor::block_on(ticket),
        Err(GraphicsCallbackHostFailure::Retired)
    );
    drop(processing);
    assert_eq!(dropped.get(), 1);
    assert_eq!(calls.get(), 0);
    assert!(release.send(()).is_err(), "abandoned turn has no receiver");
}

#[test]
fn replacement_after_callback_start_is_indeterminate_and_never_replays() {
    let queue = queue(2);
    let calls = Rc::new(Cell::new(0));
    let (release, gate) = oneshot::channel::<()>();
    let gate = Rc::new(RefCell::new(Some(gate)));
    let callback: Rc<GraphicsCallbackResponder> = {
        let calls = calls.clone();
        let gate = gate.clone();
        Rc::new(move |_| {
            calls.set(calls.get() + 1);
            let gate = gate.borrow_mut().take().expect("one callback invocation");
            Box::pin(async move {
                gate.await.expect("release callback");
                Ok(GraphicsCallbackResult::Completed)
            })
        })
    };
    let (_registration, ingress) = queue
        .register_response(selected(), 7, GraphicsCallbackKind::PanActionPre, callback)
        .unwrap();
    let ticket = ingress
        .request(event(GraphicsCallbackKind::PanActionPre))
        .unwrap();
    let mut processing = Box::pin(queue.process());
    futures::executor::block_on(futures::future::poll_fn(|context| {
        assert!(processing.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    }));
    assert_eq!(calls.get(), 1);
    assert_eq!(ingress.revoke(), GraphicsCallbackDisposition::Indeterminate);
    release.send(()).unwrap();
    futures::executor::block_on(processing).unwrap();
    assert_eq!(ticket.disposition(), GraphicsCallbackDisposition::Published);
    assert_eq!(
        futures::executor::block_on(ticket),
        Ok(GraphicsCallbackResult::Completed)
    );
    futures::executor::block_on(queue.process()).unwrap();
    assert_eq!(calls.get(), 1);
}
