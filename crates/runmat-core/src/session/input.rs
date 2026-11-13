use super::*;
use std::sync::Mutex;

impl RunMatSession {
    /// Keep session input available through the outer lifecycle drain. The
    /// source-command recorder temporarily replaces this adapter and restores
    /// it before callbacks retained past that command are driven.
    pub(super) fn install_lifecycle_input(&self) -> runmat_runtime::interaction::AsyncHandlerGuard {
        let _scope = self.runtime_context.enter();
        self.install_execution_input().0
    }

    /// Install session input in the currently entered runtime context. Both
    /// interactive source and immutable executable invocation use this adapter;
    /// dropping the guard restores the previous context-owned handler.
    pub(super) fn install_execution_input(
        &self,
    ) -> (
        runmat_runtime::interaction::AsyncHandlerGuard,
        Arc<Mutex<Vec<StdinEvent>>>,
    ) {
        let stdin_events: Arc<Mutex<Vec<StdinEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let host_async_handler = self.async_input_handler.clone();
        let stdin_events_async = Arc::clone(&stdin_events);
        let runtime_async_handler: Arc<runmat_runtime::interaction::AsyncInteractionHandler> =
            Arc::new(
                move |prompt: runmat_runtime::interaction::InteractionPromptOwned| {
                    let request_kind = match prompt.kind {
                        runmat_runtime::interaction::InteractionKind::Line { echo } => {
                            InputRequestKind::Line { echo }
                        }
                        runmat_runtime::interaction::InteractionKind::KeyPress => {
                            InputRequestKind::KeyPress
                        }
                    };
                    let request = InputRequest {
                        prompt: prompt.prompt,
                        kind: request_kind,
                    };
                    let (event_kind, echo_flag) = match &request.kind {
                        InputRequestKind::Line { echo } => (StdinEventKind::Line, *echo),
                        InputRequestKind::KeyPress => (StdinEventKind::KeyPress, false),
                    };
                    let mut event = StdinEvent {
                        prompt: request.prompt.clone(),
                        kind: event_kind,
                        echo: echo_flag,
                        value: None,
                        error: None,
                    };

                    let stdin_events_async = Arc::clone(&stdin_events_async);
                    let host_async_handler = host_async_handler.clone();
                    Box::pin(async move {
                        let resp: Result<InputResponse, String> =
                            if let Some(handler) = host_async_handler {
                                handler(request).await
                            } else {
                                match &request.kind {
                                    InputRequestKind::Line { echo } => {
                                        runmat_runtime::interaction::default_read_line(
                                            &request.prompt,
                                            *echo,
                                        )
                                        .map(InputResponse::Line)
                                    }
                                    InputRequestKind::KeyPress => {
                                        runmat_runtime::interaction::default_wait_for_key(
                                            &request.prompt,
                                        )
                                        .map(|_| InputResponse::KeyPress)
                                    }
                                }
                            };

                        let resp = resp.inspect_err(|err| {
                            event.error = Some(err.clone());
                            if let Ok(mut guard) = stdin_events_async.lock() {
                                guard.push(event.clone());
                            }
                        })?;

                        let interaction_resp = match resp {
                            InputResponse::Line(value) => {
                                event.value = Some(value.clone());
                                if let Ok(mut guard) = stdin_events_async.lock() {
                                    guard.push(event);
                                }
                                runmat_runtime::interaction::InteractionResponse::Line(value)
                            }
                            InputResponse::KeyPress => {
                                if let Ok(mut guard) = stdin_events_async.lock() {
                                    guard.push(event);
                                }
                                runmat_runtime::interaction::InteractionResponse::KeyPress
                            }
                        };
                        Ok(interaction_resp)
                    })
                },
            );
        let _async_input_guard =
            runmat_runtime::interaction::replace_async_handler(Some(runtime_async_handler));
        (_async_input_guard, stdin_events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_runtime::interaction::{
        replace_async_handler, request_line_async, wait_for_key_async, InteractionResponse,
    };

    #[test]
    fn session_input_adapter_preserves_requests_events_and_restores_context() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            session.install_async_input_handler(|request| async move {
                match request.kind {
                    InputRequestKind::Line { echo } => {
                        assert_eq!(request.prompt, "line: ");
                        assert!(!echo);
                        Ok(InputResponse::Line("value".into()))
                    }
                    InputRequestKind::KeyPress => {
                        assert_eq!(request.prompt, "key: ");
                        Ok(InputResponse::KeyPress)
                    }
                }
            });
            let runtime = session.runtime_context.clone();
            block_on(runtime.scope(async {
                let _previous = replace_async_handler(Some(Arc::new(|_| {
                    Box::pin(async { Ok(InteractionResponse::Line("previous".into())) })
                })));
                let (guard, events) = session.install_execution_input();
                assert_eq!(request_line_async("line: ", false).await.unwrap(), "value");
                wait_for_key_async("key: ").await.unwrap();
                {
                    let events = events.lock().unwrap();
                    assert_eq!(events.len(), 2);
                    assert_eq!(events[0].value.as_deref(), Some("value"));
                    assert!(!events[0].echo);
                    assert!(events[1].value.is_none());
                }
                drop(guard);
                assert_eq!(
                    request_line_async("restored: ", false).await.unwrap(),
                    "previous"
                );
            }));
        });
    }

    #[test]
    fn session_input_adapter_retains_host_error_and_event() {
        runmat_gc::gc_test_context(|| {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            session.install_async_input_handler(|_| async { Err("host input failed".into()) });
            let runtime = session.runtime_context.clone();
            block_on(runtime.scope(async {
                let (_guard, events) = session.install_execution_input();
                let error = request_line_async("error: ", true).await.unwrap_err();
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:interaction:AsyncHandlerError")
                );
                let events = events.lock().unwrap();
                assert_eq!(events[0].error.as_deref(), Some("host input failed"));
                assert!(events[0].value.is_none());
            }));
        });
    }
}
