//! One session-lane driver for script execution and idle host callbacks.
//!
//! The execution future and callback drain are never polled concurrently.
//! A drawnow drain inside execution retains its existing reentrant-call
//! diagnostics, while an idle host admission waits until that drain ends.

use super::*;
use futures::future::{poll_fn, select, Either};
use std::future::Future;
use std::task::Poll;

impl RuntimeGraphicsCallbackQueue {
    /// Drive an invocation and admitted GUI events on its originating local
    /// lane. An idle callback failure remains the original RuntimeError and
    /// stops this invocation; it is never converted to a successful host
    /// receipt. The caller owns the invocation's cancellation/drop policy.
    pub async fn drive_execution_with_host_callbacks<F: Future>(
        &self,
        execution: F,
    ) -> Result<F::Output, RuntimeError> {
        if self.0.driving.replace(true) {
            return Err(
                build_runtime_error("graphics callback session lane is already driving")
                    .with_identifier("RunMat:graphics:CallbackDriverReentrant")
                    .build(),
            );
        }
        struct DriveGuard<'a>(&'a Inner);
        impl Drop for DriveGuard<'_> {
            fn drop(&mut self) {
                self.0.driving.set(false);
            }
        }
        let _guard = DriveGuard(&self.0);
        let mut execution = Box::pin(execution);
        loop {
            if self.is_closed() {
                return Ok(execution.await);
            }
            let admission = self.wait_for_host_admission();
            futures::pin_mut!(admission);
            match select(execution.as_mut(), admission).await {
                Either::Left((result, _)) => return Ok(result),
                Either::Right((false, _)) => return Ok(execution.await),
                Either::Right((true, _)) => {}
            }

            // A host admission may wake while drawnow is already draining
            // inside the execution future. Keep that invocation as the sole
            // drain owner until it yields with draining=false. The wake is
            // retained by this loop, so work admitted just after drawnow's
            // terminal empty check still gets a later idle drain.
            while self.drain_in_progress() {
                let completed = poll_fn(|context| match execution.as_mut().poll(context) {
                    Poll::Ready(result) => Poll::Ready(Some(result)),
                    Poll::Pending if !self.drain_in_progress() => Poll::Ready(None),
                    Poll::Pending => Poll::Pending,
                })
                .await;
                if let Some(result) = completed {
                    return Ok(result);
                }
            }
            self.process().await?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::context::{RuntimeLocalTaskService, RuntimeLocalTaskSupervisor};
    use crate::execution::RuntimeExecutionService;
    use futures::channel::oneshot;
    use futures::FutureExt;
    use std::cell::Cell;
    use std::cell::RefCell;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn idle_callback_participates_in_the_same_local_task_supervisor_poll() {
        let host = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test host reactor");
        host.block_on(async {
            let selected = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
            let queue = RuntimeGraphicsCallbackQueue::new(
                NonZeroUsize::new(1).expect("nonzero pending"),
                NonZeroUsize::new(1).expect("nonzero registrations"),
            );
            let (begin, begun) = oneshot::channel::<()>();
            let (callback_done, callback_complete) = oneshot::channel::<()>();
            let (execution_done, execution_complete) = oneshot::channel::<()>();
            let task = supervisor
                .submission_port(selected.local_identity())
                .spawn_local(
                    selected.local_identity(),
                    Box::pin(async move {
                        begun.await.expect("callback starts local task");
                        callback_done.send(()).expect("callback still waiting");
                        execution_done.send(()).expect("execution still waiting");
                    }),
                )
                .expect("admit local task");
            let begin = Rc::new(RefCell::new(Some(begin)));
            let callback_complete = Rc::new(RefCell::new(Some(callback_complete)));
            let observed = Rc::new(Cell::new(false));
            let callback: Rc<GraphicsCallbackResponder> = {
                let supervisor = Rc::clone(&supervisor);
                let begin = Rc::clone(&begin);
                let callback_complete = Rc::clone(&callback_complete);
                let observed = Rc::clone(&observed);
                Rc::new(move |_| {
                    let begin = begin.borrow_mut().take().expect("one callback");
                    let done = callback_complete
                        .borrow_mut()
                        .take()
                        .expect("one completion");
                    let supervisor = Rc::clone(&supervisor);
                    let observed = Rc::clone(&observed);
                    Box::pin(async move {
                        begin.send(()).expect("local task retained");
                        supervisor
                            .participate_foreground(async {
                                done.await.expect("supervisor-owned local task progressed");
                            })
                            .await
                            .expect("callback runs inside the active supervisor poll");
                        observed.set(true);
                        Ok(GraphicsCallbackResult::Completed)
                    })
                })
            };
            let (_registration, ingress) = queue
                .register_response(selected, 9, GraphicsCallbackKind::PanActionPre, callback)
                .expect("register callback");
            ingress
                .send(GraphicsCallbackHostEvent {
                    figure: 9,
                    kind: GraphicsCallbackKind::PanActionPre,
                    position: None,
                })
                .expect("admit host event");
            supervisor
                .drive_until(queue.drive_execution_with_host_callbacks(async move {
                    execution_complete
                        .await
                        .expect("local execution completion");
                }))
                .await
                .expect("local supervisor")
                .expect("idle graphics driver");
            assert!(observed.get());
            assert_eq!(
                task.outcome(),
                Some(crate::context::RuntimeLocalTaskOutcome::Completed)
            );
        });
    }

    #[test]
    fn idle_host_event_runs_once_while_execution_is_pending() {
        futures::executor::block_on(async {
            let queue = RuntimeGraphicsCallbackQueue::new(
                NonZeroUsize::new(2).expect("nonzero pending"),
                NonZeroUsize::new(1).expect("nonzero registrations"),
            );
            let entered = Rc::new(Cell::new(0));
            let observed = Rc::clone(&entered);
            let callback: Rc<GraphicsCallbackResponder> = Rc::new(move |_| {
                observed.set(observed.get() + 1);
                Box::pin(async { Ok(GraphicsCallbackResult::Completed) })
            });
            let (_registration, ingress) = queue
                .register_response(
                    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())),
                    9,
                    GraphicsCallbackKind::PanActionPre,
                    callback,
                )
                .expect("register callback");
            let (complete, execution) = oneshot::channel::<()>();
            let mut lane = Box::pin(queue.drive_execution_with_host_callbacks(async move {
                execution.await.expect("execution completion")
            }));
            assert!(lane.as_mut().now_or_never().is_none());
            ingress
                .send(GraphicsCallbackHostEvent {
                    figure: 9,
                    kind: GraphicsCallbackKind::PanActionPre,
                    position: None,
                })
                .expect("admit host event");
            assert!(lane.as_mut().now_or_never().is_none());
            assert_eq!(entered.get(), 1);
            complete.send(()).expect("complete execution");
            lane.await.expect("callback driver");
            assert_eq!(entered.get(), 1);
        });
    }

    #[test]
    fn host_wake_during_drawnow_drain_does_not_start_a_second_drain() {
        futures::executor::block_on(async {
            let queue = RuntimeGraphicsCallbackQueue::new(
                NonZeroUsize::new(2).expect("nonzero pending"),
                NonZeroUsize::new(1).expect("nonzero registrations"),
            );
            let (release, suspended) = oneshot::channel::<()>();
            let first = Rc::new(RefCell::new(Some(suspended)));
            let calls = Rc::new(Cell::new(0));
            let callback: Rc<GraphicsCallbackResponder> = {
                let first = Rc::clone(&first);
                let calls = Rc::clone(&calls);
                Rc::new(move |_| {
                    calls.set(calls.get() + 1);
                    let gate = first.borrow_mut().take();
                    Box::pin(async move {
                        if let Some(gate) = gate {
                            gate.await.expect("release first callback");
                        }
                        Ok(GraphicsCallbackResult::Completed)
                    })
                })
            };
            let (_registration, ingress) = queue
                .register_response(
                    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())),
                    9,
                    GraphicsCallbackKind::PanActionPre,
                    callback,
                )
                .expect("register callback");
            ingress
                .send(GraphicsCallbackHostEvent {
                    figure: 9,
                    kind: GraphicsCallbackKind::PanActionPre,
                    position: None,
                })
                .expect("first event");
            let mut lane = Box::pin(
                queue.drive_execution_with_host_callbacks(async { queue.process().await }),
            );
            assert!(lane.as_mut().now_or_never().is_none());
            assert_eq!(calls.get(), 1);
            assert!(queue.drain_in_progress());
            ingress
                .send(GraphicsCallbackHostEvent {
                    figure: 9,
                    kind: GraphicsCallbackKind::PanActionPre,
                    position: None,
                })
                .expect("second event during first callback");
            assert!(lane.as_mut().now_or_never().is_none());
            assert_eq!(calls.get(), 1, "second drain must not race drawnow");
            release.send(()).expect("release first callback");
            lane.await
                .expect("session driver")
                .expect("one ordered callback drain");
            assert_eq!(calls.get(), 2);
        });
    }

    #[test]
    fn second_session_lane_driver_rejects_until_first_is_dropped() {
        futures::executor::block_on(async {
            let queue = RuntimeGraphicsCallbackQueue::new(
                NonZeroUsize::new(1).expect("nonzero pending"),
                NonZeroUsize::new(1).expect("nonzero registrations"),
            );
            let mut first = Box::pin(
                queue.drive_execution_with_host_callbacks(futures::future::pending::<()>()),
            );
            assert!(first.as_mut().now_or_never().is_none());
            let rejected = queue
                .drive_execution_with_host_callbacks(async {})
                .await
                .expect_err("one session owns the callback pump");
            assert_eq!(
                rejected.identifier(),
                Some("RunMat:graphics:CallbackDriverReentrant")
            );
            drop(first);
            queue
                .drive_execution_with_host_callbacks(async {})
                .await
                .expect("dropped driver releases session lane");
        });
    }
}
