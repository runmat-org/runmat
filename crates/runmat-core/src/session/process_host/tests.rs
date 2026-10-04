#![cfg(unix)]

use std::ffi::OsString;
use std::future::Future;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crate::session::RunMatSession;

static MAIL_ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());

struct MailEnvironment {
    command: Option<OsString>,
    outbox: Option<OsString>,
    _lock: MutexGuard<'static, ()>,
}

impl MailEnvironment {
    fn fake_command(executable: &Path) -> Self {
        let lock = MAIL_ENVIRONMENT_LOCK.lock().expect("mail environment lock");
        let command = std::env::var_os("RUNMAT_SENDMAIL_COMMAND");
        let outbox = std::env::var_os("RUNMAT_SENDMAIL_OUTBOX");
        std::env::set_var("RUNMAT_SENDMAIL_COMMAND", executable);
        std::env::remove_var("RUNMAT_SENDMAIL_OUTBOX");
        Self {
            command,
            outbox,
            _lock: lock,
        }
    }
}

impl Drop for MailEnvironment {
    fn drop(&mut self) {
        match &self.command {
            Some(value) => std::env::set_var("RUNMAT_SENDMAIL_COMMAND", value),
            None => std::env::remove_var("RUNMAT_SENDMAIL_COMMAND"),
        }
        match &self.outbox {
            Some(value) => std::env::set_var("RUNMAT_SENDMAIL_OUTBOX", value),
            None => std::env::remove_var("RUNMAT_SENDMAIL_OUTBOX"),
        }
    }
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

fn execute(session: &mut RunMatSession, source: &str) -> Result<(), crate::RunError> {
    let request = request(session, source);
    futures::executor::block_on(session.execute_request(request))
        .result
        .map(|_| ())
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
    let _mail_environment = MailEnvironment::fake_command(&mail);
    let mut session = isolated_session();
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
    assert!(
        response.result.is_err(),
        "cancelled call cannot publish success"
    );
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
    execute(
        &mut session,
        &format!(
            "[code, output] = system('{}'); assert(code == 0); assert(contains(output, 'owned-system')); assert(contains(output, 'stderr'));",
            matlab_literal(&system),
        ),
    )
    .expect("system uses the installed non-Tokio Core process host");

    let system_failed = temporary.path().join("fake-system-fail");
    executable(&system_failed, "printf 'system-error' >&2; exit 23");
    execute(
        &mut session,
        &format!(
            "[code, output] = system('{}'); assert(code == 23); assert(contains(output, 'system-error'));",
            matlab_literal(&system_failed),
        ),
    )
    .expect("nonzero system exit is returned, not erased");

    let mail = temporary.path().join("fake-sendmail");
    let body = temporary.path().join("captured.eml");
    executable(&mail, &format!("cat > {}", shell_literal(&body)));
    let _mail_environment = MailEnvironment::fake_command(&mail);
    execute(
        &mut session,
        "status = sendmail('alice@example.test', 'Subject', 'Body'); assert(status == 0);",
    )
    .expect("sendmail uses the installed owned-input process host");
    assert_eq!(
        std::fs::read(&body).expect("fake command captured body"),
        b"To: alice@example.test\nSubject: Subject\nContent-Type: text/plain; charset=utf-8\n\nBody\n"
    );

    let failed = temporary.path().join("fake-sendmail-fail");
    executable(&failed, "printf 'declined' >&2; exit 17");
    drop(_mail_environment);
    let _mail_environment = MailEnvironment::fake_command(&failed);
    let error = execute(
        &mut session,
        "sendmail('alice@example.test', 'Subject', 'Body');",
    )
    .expect_err("nonzero local mail exit must fail public call");
    let crate::RunError::Runtime(error) = error else {
        panic!("expected runtime mail error");
    };
    assert_eq!(error.identifier(), Some("RunMat:sendmail:CommandFailure"));
    assert!(error.message().contains("declined"));
}
