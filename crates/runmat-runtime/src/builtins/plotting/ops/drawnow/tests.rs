use super::*;
use crate::context::{
    GraphicsCallbackReceipt, GraphicsFlushReceipt, RuntimeGraphicsFlushOperation,
    RuntimeGraphicsPresentationService, RuntimeLanguageMode, RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

struct ReadyFlush {
    receipt: GraphicsFlushReceipt,
    published: bool,
}

impl Future for ReadyFlush {
    type Output = BuiltinResult<GraphicsFlushReceipt>;

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.published = true;
        Poll::Ready(Ok(self.receipt))
    }
}

impl RuntimeGraphicsFlushOperation for ReadyFlush {
    fn disposition(&self) -> GraphicsFlushDisposition {
        if self.published {
            GraphicsFlushDisposition::Published
        } else {
            GraphicsFlushDisposition::NotPublished
        }
    }

    fn abandon(self: Pin<&mut Self>) -> GraphicsFlushDisposition {
        self.as_ref().get_ref().disposition()
    }
}

type GraphicsRequestLog = Rc<RefCell<Vec<GraphicsFlushRequest>>>;

struct TestGraphics {
    requests: GraphicsRequestLog,
    calls: Rc<Cell<usize>>,
    receipt: GraphicsFlushReceipt,
}

impl RuntimeGraphicsPresentationService for TestGraphics {
    fn flush(&self, request: GraphicsFlushRequest) -> Pin<Box<dyn RuntimeGraphicsFlushOperation>> {
        self.calls.set(self.calls.get() + 1);
        self.requests.borrow_mut().push(request);
        Box::pin(ReadyFlush {
            receipt: self.receipt,
            published: false,
        })
    }
}

fn selected_runtime(
    receipt: GraphicsFlushReceipt,
) -> (RuntimeContext, GraphicsRequestLog, Rc<Cell<usize>>) {
    let requests = Rc::new(RefCell::new(Vec::new()));
    let calls = Rc::new(Cell::new(0));
    let service = Rc::new(TestGraphics {
        requests: requests.clone(),
        calls: calls.clone(),
        receipt,
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_graphics_presentation(service));
    runtime.set_language_mode(RuntimeLanguageMode::RunMat);
    runtime.set_runmat_extensions_enabled(true);
    (runtime, requests, calls)
}

#[test]
fn all_controls_preserve_distinct_rate_and_callback_requests() {
    for (tokens, expected) in [
        (vec![], request_for(DrawNowControl::Full)),
        (vec!["limitrate"], request_for(DrawNowControl::LimitRate)),
        (
            vec!["nocallbacks"],
            request_for(DrawNowControl::NoCallbacks),
        ),
        (vec!["update"], request_for(DrawNowControl::Update)),
        (vec!["expose"], request_for(DrawNowControl::Expose)),
        (
            vec!["limitrate", "nocallbacks"],
            request_for(DrawNowControl::LimitRateNoCallbacks),
        ),
    ] {
        let (runtime, requests, _) = selected_runtime(GraphicsFlushReceipt {
            presentation: GraphicsPresentation::Presented,
            callbacks: match expected.callbacks {
                GraphicsCallbackPolicy::Process => GraphicsCallbackReceipt::Processed,
                GraphicsCallbackPolicy::Defer => GraphicsCallbackReceipt::Deferred,
            },
        });
        let arguments = crate::builtins::common::test_support::builtin_arguments(
            tokens.into_iter().map(Value::from).collect(),
        );
        assert_eq!(
            block_on(drawnow_builtin(Some(runtime), Some(0), arguments)).unwrap(),
            Value::output_list(Vec::new())
        );
        assert_eq!(requests.borrow().as_slice(), &[expected]);
    }
}

#[test]
fn no_figure_still_requires_processed_callbacks_and_no_service_is_not_success() {
    let (runtime, requests, _) = selected_runtime(GraphicsFlushReceipt {
        presentation: GraphicsPresentation::NoFigure,
        callbacks: GraphicsCallbackReceipt::Processed,
    });
    assert_eq!(
        block_on(drawnow_builtin(
            Some(runtime),
            Some(0),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .unwrap(),
        Value::output_list(Vec::new())
    );
    assert_eq!(requests.borrow().len(), 1);

    let absent = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    assert_eq!(
        block_on(drawnow_builtin(
            Some(absent),
            Some(0),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .unwrap_err()
        .identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );
}

#[test]
fn output_and_invalid_controls_precede_host_submission() {
    let (runtime, _, calls) = selected_runtime(GraphicsFlushReceipt {
        presentation: GraphicsPresentation::NoFigure,
        callbacks: GraphicsCallbackReceipt::Processed,
    });
    assert_eq!(
        block_on(drawnow_builtin(
            Some(runtime.clone()),
            Some(2),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .unwrap_err()
        .identifier(),
        DRAWNOW_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    assert_eq!(
        block_on(drawnow_builtin(
            Some(runtime),
            Some(0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("bogus")])
        ))
        .unwrap_err()
        .identifier(),
        Some("RunMat:drawnow:InvalidControl")
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn unpolled_public_future_does_not_submit_and_status_is_explicit() {
    let (runtime, _, calls) = selected_runtime(GraphicsFlushReceipt {
        presentation: GraphicsPresentation::NoFigure,
        callbacks: GraphicsCallbackReceipt::Processed,
    });
    let future = drawnow_builtin(
        Some(runtime.clone()),
        Some(0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    );
    drop(future);
    assert_eq!(calls.get(), 0);

    assert_eq!(
        block_on(drawnow_builtin(
            Some(runtime),
            Some(1),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .unwrap(),
        Value::Bool(true)
    );
    assert_eq!(calls.get(), 1);
}

fn effect_disposition(error: &RuntimeError) -> GraphicsFlushDisposition {
    error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<errors::GraphicsFlushEffectSource>())
        .expect("typed graphics effect source")
        .disposition()
}

#[test]
fn host_callback_policy_violations_preserve_published_effect_cause() {
    for (control, callbacks) in [
        ("nocallbacks", GraphicsCallbackReceipt::Processed),
        ("limitrate", GraphicsCallbackReceipt::Deferred),
    ] {
        let (runtime, _, calls) = selected_runtime(GraphicsFlushReceipt {
            presentation: GraphicsPresentation::Presented,
            callbacks,
        });
        let error = block_on(drawnow_builtin(
            Some(runtime),
            Some(0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(control)]),
        ))
        .expect_err("wrong callback receipt must fail closed");
        assert_eq!(
            error.identifier(),
            runmat_builtins::DRAWNOW_ERROR_PRESENTATION.identifier
        );
        assert_eq!(
            effect_disposition(&error),
            GraphicsFlushDisposition::Published
        );
        assert_eq!(calls.get(), 1);
    }
}

#[derive(Default)]
struct PendingState {
    polls: Cell<usize>,
    drops: Cell<usize>,
    abandoned: Cell<bool>,
}

struct PendingFlush {
    state: Rc<PendingState>,
    after_poll: GraphicsFlushDisposition,
}

impl Future for PendingFlush {
    type Output = BuiltinResult<GraphicsFlushReceipt>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.state.polls.set(self.state.polls.get() + 1);
        Poll::Pending
    }
}

impl RuntimeGraphicsFlushOperation for PendingFlush {
    fn disposition(&self) -> GraphicsFlushDisposition {
        if self.state.polls.get() == 0 {
            GraphicsFlushDisposition::NotPublished
        } else {
            self.after_poll
        }
    }

    fn abandon(self: Pin<&mut Self>) -> GraphicsFlushDisposition {
        let this = self.as_ref().get_ref();
        this.state.abandoned.set(true);
        this.disposition()
    }
}

impl Drop for PendingFlush {
    fn drop(&mut self) {
        self.state.drops.set(self.state.drops.get() + 1);
    }
}

struct PendingGraphics {
    state: Rc<PendingState>,
    disposition: GraphicsFlushDisposition,
}

impl RuntimeGraphicsPresentationService for PendingGraphics {
    fn flush(&self, _: GraphicsFlushRequest) -> Pin<Box<dyn RuntimeGraphicsFlushOperation>> {
        Box::pin(PendingFlush {
            state: self.state.clone(),
            after_poll: self.disposition,
        })
    }
}

#[test]
fn pending_cancellation_preserves_exact_effect_disposition_and_retires_operation() {
    for disposition in [
        GraphicsFlushDisposition::NotPublished,
        GraphicsFlushDisposition::Indeterminate,
    ] {
        let cancellation = CancellationSource::new();
        let state = Rc::new(PendingState::default());
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        )
        .with_service_ports(
            RuntimeServicePorts::default().with_graphics_presentation(Rc::new(PendingGraphics {
                state: state.clone(),
                disposition,
            })),
        );
        let mut call = Box::pin(drawnow_builtin(
            Some(runtime),
            Some(0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(call.as_mut().poll(&mut context).is_pending());
        assert_eq!(state.polls.get(), 1);
        cancellation.cancel(CancellationReason::User);
        let Poll::Ready(Err(error)) = call.as_mut().poll(&mut context) else {
            panic!("selected cancellation must stop a pending host flush")
        };
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(effect_disposition(&error), disposition);
        assert_eq!(state.polls.get(), 1, "no host replay after cancellation");
        assert!(state.abandoned.get());
        assert_eq!(state.drops.get(), 1);
    }
}

struct TerminalFlush {
    result: Option<BuiltinResult<GraphicsFlushReceipt>>,
    cancellation: Option<CancellationSource>,
    after_poll: GraphicsFlushDisposition,
    polled: bool,
}

impl Future for TerminalFlush {
    type Output = BuiltinResult<GraphicsFlushReceipt>;

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polled = true;
        if let Some(cancellation) = self.cancellation.take() {
            cancellation.cancel(CancellationReason::User);
        }
        Poll::Ready(
            self.result
                .take()
                .expect("terminal host result polled once"),
        )
    }
}

impl RuntimeGraphicsFlushOperation for TerminalFlush {
    fn disposition(&self) -> GraphicsFlushDisposition {
        if self.polled {
            self.after_poll
        } else {
            GraphicsFlushDisposition::NotPublished
        }
    }

    fn abandon(self: Pin<&mut Self>) -> GraphicsFlushDisposition {
        self.as_ref().get_ref().disposition()
    }
}

struct TerminalGraphics {
    result: RefCell<Option<BuiltinResult<GraphicsFlushReceipt>>>,
    cancellation: Option<CancellationSource>,
    disposition: GraphicsFlushDisposition,
}

impl RuntimeGraphicsPresentationService for TerminalGraphics {
    fn flush(&self, _: GraphicsFlushRequest) -> Pin<Box<dyn RuntimeGraphicsFlushOperation>> {
        Box::pin(TerminalFlush {
            result: self.result.borrow_mut().take(),
            cancellation: self.cancellation.clone(),
            after_poll: self.disposition,
            polled: false,
        })
    }
}

fn terminal_runtime(
    result: BuiltinResult<GraphicsFlushReceipt>,
    disposition: GraphicsFlushDisposition,
    cancellation: Option<CancellationSource>,
) -> RuntimeContext {
    let runtime = match &cancellation {
        Some(source) => RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        ),
        None => RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new())),
    };
    runtime.with_service_ports(
        RuntimeServicePorts::default().with_graphics_presentation(Rc::new(TerminalGraphics {
            result: RefCell::new(Some(result)),
            cancellation,
            disposition,
        })),
    )
}

#[test]
fn terminal_host_error_keeps_original_diagnostic_and_published_effect() {
    let original = crate::build_runtime_error("host frame failed")
        .with_identifier("Host:FrameFailure")
        .build();
    let runtime = terminal_runtime(Err(original), GraphicsFlushDisposition::Published, None);
    let error = block_on(drawnow_builtin(
        Some(runtime),
        Some(0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("Host:FrameFailure"));
    assert_eq!(error.message, "host frame failed");
    assert_eq!(
        effect_disposition(&error),
        GraphicsFlushDisposition::Published
    );
}

#[test]
fn invalid_published_receipt_preserves_effect_without_replay() {
    let runtime = terminal_runtime(
        Ok(GraphicsFlushReceipt {
            presentation: GraphicsPresentation::DeferredByRate,
            callbacks: GraphicsCallbackReceipt::Processed,
        }),
        GraphicsFlushDisposition::Published,
        None,
    );
    let error = block_on(drawnow_builtin(
        Some(runtime),
        Some(0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::DRAWNOW_ERROR_PRESENTATION.identifier
    );
    assert_eq!(
        effect_disposition(&error),
        GraphicsFlushDisposition::Published
    );

    let runtime = terminal_runtime(
        Ok(GraphicsFlushReceipt {
            presentation: GraphicsPresentation::DeferredByRate,
            callbacks: GraphicsCallbackReceipt::Deferred,
        }),
        GraphicsFlushDisposition::Published,
        None,
    );
    let error = block_on(drawnow_builtin(
        Some(runtime),
        Some(0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("update")]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::DRAWNOW_ERROR_PRESENTATION.identifier
    );
    assert_eq!(
        effect_disposition(&error),
        GraphicsFlushDisposition::Published
    );
}

#[test]
fn update_busy_deferral_is_distinct_from_rate_limiting() {
    let runtime = terminal_runtime(
        Ok(GraphicsFlushReceipt {
            presentation: GraphicsPresentation::DeferredByBusy,
            callbacks: GraphicsCallbackReceipt::Deferred,
        }),
        GraphicsFlushDisposition::NotPublished,
        None,
    );
    assert_eq!(
        block_on(drawnow_builtin(
            Some(runtime),
            Some(0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("update")])
        ))
        .unwrap(),
        Value::output_list(Vec::new())
    );
}

#[test]
fn terminal_success_and_error_win_same_poll_cancellation() {
    for result in [
        Ok(GraphicsFlushReceipt {
            presentation: GraphicsPresentation::Presented,
            callbacks: GraphicsCallbackReceipt::Processed,
        }),
        Err(crate::build_runtime_error("terminal host error")
            .with_identifier("Host:Terminal")
            .build()),
    ] {
        let cancellation = CancellationSource::new();
        let runtime = terminal_runtime(
            result,
            GraphicsFlushDisposition::Published,
            Some(cancellation.clone()),
        );
        let outcome = block_on(drawnow_builtin(
            Some(runtime),
            Some(0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ));
        assert!(cancellation.token().is_cancelled());
        match outcome {
            Ok(value) => assert_eq!(value, Value::output_list(Vec::new())),
            Err(error) => {
                assert_eq!(error.identifier(), Some("Host:Terminal"));
                assert_eq!(
                    effect_disposition(&error),
                    GraphicsFlushDisposition::Published
                );
            }
        }
    }
}
