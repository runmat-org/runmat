#![cfg(unix)]

use std::future::Future;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crate::session::RunMatSession;

fn install_mail_command(session: &RunMatSession, executable: &Path) {
    let mut environment = session
        .environment_snapshot()
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    environment.insert(
        "RUNMAT_SENDMAIL_COMMAND".into(),
        executable.to_str().unwrap().into(),
    );
    environment.remove("RUNMAT_SENDMAIL_OUTBOX");
    session
        .runtime_context
        .install_environment_snapshot(environment.into_iter().collect())
        .expect("install session mail environment");
}

struct RestoreDirectory(PathBuf);

impl Drop for RestoreDirectory {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.0).expect("restore test working directory");
    }
}

fn isolated_session() -> RunMatSession {
    let mut session = RunMatSession::with_options(false, false).expect("native Core session");
    session
        .install_application_settings(Rc::new(
            runmat_runtime::context::IsolatedMemoryPreferenceService::new(),
        ))
        .expect("isolated application settings");
    session
}

fn execute(
    session: &mut RunMatSession,
    source: &str,
) -> Result<runmat_runtime::execution::RootedResult<crate::abi::ExecutionOutcome>, crate::RunError>
{
    let request = request(session, source);
    futures::executor::block_on(session.execute_request(request)).result
}

fn execute_success(session: &mut RunMatSession, source: &str) {
    let outcome = execute(session, source).expect("Core execution outcome");
    assert!(
        outcome
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity != crate::abi::DiagnosticSeverity::Error),
        "unexpected runtime diagnostics: {:?}",
        outcome.diagnostics
    );
}

fn request(session: &RunMatSession, source: &str) -> crate::abi::ExecutionRequest {
    crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<process-host-test>".into(),
            text: source.into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    )
}

#[test]
fn ordinary_executor_cancellation_drops_owned_mail_child() {
    assert!(tokio::runtime::Handle::try_current().is_err());
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::tempdir().expect("isolated process directory");
    let previous = std::env::current_dir().expect("original directory");
    let _restore = RestoreDirectory(previous);
    std::env::set_current_dir(temporary.path()).expect("isolate process directory");
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::NativeFsProvider,
    ));
    let marker = temporary.path().join("pending-child-pid");
    let mail = temporary.path().join("pending-sendmail");
    executable(
        &mail,
        &format!(
            "printf '%s' \"$$\" > {}; cat >/dev/null; exec /bin/sleep 30",
            shell_literal(&marker)
        ),
    );
    let mut session = isolated_session();
    install_mail_command(&session, &mail);
    let interrupt = session.interrupt_handle();
    let request = request(
        &session,
        "sendmail('alice@example.test', 'Subject', 'Body');",
    );
    let mut call = Box::pin(session.execute_request(request));
    let deadline = Instant::now() + Duration::from_secs(5);
    let pid = loop {
        if let Ok(text) = std::fs::read_to_string(&marker) {
            if let Ok(pid) = text.parse::<u32>() {
                break pid;
            }
        }
        assert!(Instant::now() < deadline, "fake mail child must start");
        assert!(matches!(
            call.as_mut()
                .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
            Poll::Pending
        ));
        std::thread::sleep(Duration::from_millis(5));
    };
    interrupt.cancel(runmat_execution::CancellationReason::User);
    let response = futures::executor::block_on(call);
    if let Ok(outcome) = response.result {
        assert!(
            outcome
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.severity == crate::abi::DiagnosticSeverity::Error),
            "cancelled call cannot publish a successful outcome"
        );
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while runmat_process_host::is_process_alive(pid) {
        assert!(Instant::now() < deadline, "owned mail child must terminate");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn executable(path: &Path, body: &str) {
    std::fs::write(path, format!("#!/bin/sh\n{body}\n")).expect("write fake executable");
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .expect("mark fake executable");
}

fn shell_literal(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

fn matlab_literal(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

#[test]
fn ordinary_executor_runs_system_and_mail_through_core_host_driver() {
    assert!(tokio::runtime::Handle::try_current().is_err());
    let _serial = runmat_filesystem::provider_override_lock();
    let temporary = tempfile::tempdir().expect("isolated process directory");
    let previous = std::env::current_dir().expect("original directory");
    let _restore = RestoreDirectory(previous);
    std::env::set_current_dir(temporary.path()).expect("isolate process directory");
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::NativeFsProvider,
    ));

    let system = temporary.path().join("fake-system");
    executable(&system, "printf 'owned-system'; printf ' stderr' >&2");
    let mut session = isolated_session();
    execute_success(
        &mut session,
        &format!(
            "[code, output] = system('{}'); assert(code == 0); assert(contains(output, 'owned-system')); assert(contains(output, 'stderr'));",
            matlab_literal(&system),
        ),
    );

    let system_failed = temporary.path().join("fake-system-fail");
    executable(&system_failed, "printf 'system-error' >&2; exit 23");
    execute_success(
        &mut session,
        &format!(
            "[code, output] = system('{}'); assert(code == 23); assert(contains(output, 'system-error'));",
            matlab_literal(&system_failed),
        ),
    );

    let mail = temporary.path().join("fake-sendmail");
    let body = temporary.path().join("captured.eml");
    executable(&mail, &format!("cat > {}", shell_literal(&body)));
    install_mail_command(&session, &mail);
    execute_success(
        &mut session,
        "status = sendmail('alice@example.test', 'Subject', 'Body'); assert(status == 0);",
    );
    assert_eq!(
        std::fs::read(&body).expect("fake command captured body"),
        b"To: alice@example.test\nSubject: Subject\nContent-Type: text/plain; charset=utf-8\n\nBody\n"
    );

    let failed = temporary.path().join("fake-sendmail-fail");
    executable(&failed, "printf 'declined' >&2; exit 17");
    install_mail_command(&session, &failed);
    let outcome = execute(
        &mut session,
        "sendmail('alice@example.test', 'Subject', 'Body');",
    )
    .expect("runtime mail failure is an outcome diagnostic");
    let error = outcome
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.severity == crate::abi::DiagnosticSeverity::Error)
        .expect("nonzero local mail exit must fail public call");
    assert_eq!(error.code, "RunMat:sendmail:CommandFailure");
    assert!(error.message.contains("declined"));
}
