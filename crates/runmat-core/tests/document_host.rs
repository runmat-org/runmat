#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use runmat_core::{execute_text_request_for_testing, RunMatSession};
use runmat_runtime::{
    console::ConsoleStream,
    context::{HostDocumentRequest, HostInteraction, RuntimeHostService, RuntimeServiceFuture},
    interaction::InteractionResponse,
    warning_store::RuntimeWarning,
    RuntimeError,
};

#[derive(Default)]
struct Host {
    documents: RefCell<Vec<HostDocumentRequest>>,
    console: RefCell<Vec<runmat_runtime::console::ConsoleText>>,
    warnings: RefCell<Vec<RuntimeWarning>>,
}

impl RuntimeHostService for Host {
    fn console(&self, _: ConsoleStream, text: runmat_runtime::console::ConsoleText) {
        self.console.borrow_mut().push(text);
    }
    fn warning(&self, warning: RuntimeWarning) {
        self.warnings.borrow_mut().push(warning);
    }
    fn interact(
        &self,
        request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        if let Some(future) = runmat_runtime::interaction::invoke_installed_async_handler(request) {
            return future;
        }
        Box::pin(async { Ok(InteractionResponse::Line("accepted".into())) })
    }
    fn open_document(
        &self,
        request: HostDocumentRequest,
    ) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        self.documents.borrow_mut().push(request);
        Box::pin(async { Ok(()) })
    }
}

#[test]
fn installed_document_hosts_are_session_local_and_reach_actual_open() {
    runmat_gc::gc_test_context(|| {
        let first = Rc::new(Host::default());
        let second = Rc::new(Host::default());
        let mut one = RunMatSession::with_options(false, false).unwrap();
        let mut two = RunMatSession::with_options(false, false).unwrap();
        for (session, host) in [(&mut one, first.clone()), (&mut two, second.clone())] {
            session.set_desktop_host_available(true);
            session.install_host_service(host);
        }
        let result =
            execute_text_request_for_testing(&mut one, "alpha = 1; open('alpha');").unwrap();
        assert!(result.error.is_none(), "{:?}", result.error);
        assert_eq!(
            *first.documents.borrow(),
            vec![HostDocumentRequest::OpenVariable {
                name: "alpha".into()
            }]
        );
        assert!(second.documents.borrow().is_empty());
        let result = execute_text_request_for_testing(&mut two, "beta = 2; open('beta');").unwrap();
        assert!(result.error.is_none(), "{:?}", result.error);
        assert_eq!(
            *second.documents.borrow(),
            vec![HostDocumentRequest::OpenVariable {
                name: "beta".into()
            }]
        );
        assert_eq!(first.documents.borrow().len(), 1);
    });
}

#[test]
fn desktop_capability_without_document_host_does_not_report_success() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        session.set_desktop_host_available(true);
        let result = execute_text_request_for_testing(&mut session, "answer = 42; open('answer');");
        assert!(
            match result {
                Err(error) =>
                    error.to_string().contains("document") || error.to_string().contains("editor"),
                Ok(result) => result.error.is_some(),
            },
            "a capability declaration must not substitute for an actual host"
        );
    });
}

#[test]
fn installed_host_keeps_core_stdin_event_recording() {
    struct Interactive;
    impl Drop for Interactive {
        fn drop(&mut self) {
            runmat_runtime::interaction::force_interactive_stdin_for_tests(false);
        }
    }
    runmat_gc::gc_test_context(|| {
        runmat_runtime::interaction::force_interactive_stdin_for_tests(true);
        let _interactive = Interactive;
        let mut session = RunMatSession::with_options(false, false).unwrap();
        session.install_host_service(Rc::new(Host::default()));
        session.install_async_input_handler(|_| async {
            Ok(runmat_core::InputResponse::Line("answer".into()))
        });
        let result =
            execute_text_request_for_testing(&mut session, "name = input('Name: ', 's');").unwrap();
        assert!(result.error.is_none(), "{:?}", result.error);
        assert_eq!(result.stdin_events.len(), 1);
        assert_eq!(result.stdin_events[0].prompt, "Name: ");
        assert_eq!(result.stdin_events[0].value.as_deref(), Some("answer"));
    });
}
