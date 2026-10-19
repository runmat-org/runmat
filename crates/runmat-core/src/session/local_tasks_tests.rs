use super::*;
use runmat_runtime::context::IsolatedMemoryPreferenceService;
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn public_execution_drives_local_callback_without_a_tokio_host() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session
        .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
        .expect("isolated settings");
    let owner = session.runtime_context.local_identity();
    let observed = Rc::new(Cell::new(false));
    let callback_observed = observed.clone();
    let callback_driver = Rc::downgrade(&session.local_task_driver());
    let lease = session
        .runtime_context
        .service_ports()
        .require_local_tasks("Core foreground callback witness")
        .expect("session local task port")
        .spawn_local(
            owner,
            Box::pin(async move {
                tokio::task::yield_now().await;
                let driver = callback_driver.upgrade().expect("Core owns local driver");
                driver
                    .participate_foreground(async {})
                    .await
                    .expect("callback is polled inside its Core driver");
                callback_observed.set(true);
            }),
        )
        .expect("schedule local callback");
    let request = crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<local-task-foreground-test>".into(),
            text: "pause(0.02);".into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let response = futures::executor::block_on(session.execute_request(request));
    let outcome = response.result.expect("public execution");
    assert!(
        outcome
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
        "unexpected diagnostics: {:?}",
        outcome.diagnostics
    );
    assert!(
        observed.get(),
        "callback progressed during public foreground execution"
    );
    assert_eq!(
        lease.outcome(),
        Some(runmat_runtime::context::RuntimeLocalTaskOutcome::Completed)
    );
}
