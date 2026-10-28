//! The selected local mail command runs through the invocation's process owner.

use std::path::PathBuf;

use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::HostCommand;

use super::{sendmail_error, sendmail_error_with_source, BuiltinResult, SendmailFailure};
use crate::context::process_invocation::{self, ProcessInvocationError};
use crate::context::RuntimeContext;

pub(super) async fn run(
    runtime: Option<&RuntimeContext>,
    executable: PathBuf,
    executable_authority: Option<&runmat_filesystem::CapturedNativePath>,
    recipients: &[String],
    message: String,
) -> BuiltinResult<()> {
    let mut command = HostCommand::new(executable.clone());
    command.arguments = vec!["-t".into(), "-oi".into()];
    command.environment_policy = EnvironmentPolicy::Inherit;
    let output = process_invocation::output_complete(
        runtime,
        command,
        Some(message.into_bytes()),
        executable_authority,
    )
    .await
    .map_err(mail_process_error)?;
    if output.exit.success {
        return Ok(());
    }
    let detail = String::from_utf8_lossy(&output.stderr);
    Err(sendmail_error(
        SendmailFailure::Command,
        format!(
            "sendmail: {} failed for {} ({})",
            executable.display(),
            recipients.join(", "),
            detail.trim()
        ),
    ))
}

fn mail_process_error(source: ProcessInvocationError) -> crate::RuntimeError {
    match source {
        ProcessInvocationError::ResourceEvidence => crate::build_runtime_error(
            "sendmail: complete command output has no certified finite resource bound",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .with_builtin("sendmail")
        .build(),
        ProcessInvocationError::Capability(source) => source.into_runtime_error(),
        ProcessInvocationError::Execution(source) => source.into_runtime_error(),
        ProcessInvocationError::DirectoryCapture(source)
        | ProcessInvocationError::StaleDirectory(source) => sendmail_error_with_source(
            SendmailFailure::Command,
            format!("sendmail: process directory unavailable: {source}"),
            source,
        ),
        ProcessInvocationError::Host(source) => sendmail_error_with_source(
            SendmailFailure::Command,
            format!("sendmail: {source}"),
            source,
        ),
        ProcessInvocationError::NoNativeDirectory | ProcessInvocationError::NoRevocation => {
            sendmail_error(SendmailFailure::Command, format!("sendmail: {source}"))
        }
        ProcessInvocationError::ForeignExecutableOwner
        | ProcessInvocationError::WrongExecutableGrant
        | ProcessInvocationError::NoExecutableRevocation => {
            sendmail_error(SendmailFailure::Command, format!("sendmail: {source}"))
        }
        ProcessInvocationError::StaleExecutable(source) => sendmail_error_with_source(
            SendmailFailure::Command,
            format!("sendmail: local executable authority revoked: {source}"),
            source,
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    use runmat_process_host::{CompleteProcessOutput, ProcessExit, ProcessHostResult};

    use super::*;
    use crate::context::{RuntimeProcessService, RuntimeServiceFuture, RuntimeServicePorts};

    struct RecordingProcess {
        calls: Rc<Cell<usize>>,
        success: bool,
    }

    impl RuntimeProcessService for RecordingProcess {
        fn output_complete(
            &self,
            _command: HostCommand,
        ) -> RuntimeServiceFuture<ProcessHostResult<CompleteProcessOutput>> {
            Box::pin(async { panic!("sendmail must use the owned-input operation") })
        }

        fn output_complete_with_input(
            &self,
            command: HostCommand,
            input: Vec<u8>,
        ) -> RuntimeServiceFuture<ProcessHostResult<CompleteProcessOutput>> {
            let calls = self.calls.clone();
            let success = self.success;
            Box::pin(async move {
                calls.set(calls.get() + 1);
                assert_eq!(command.arguments, ["-t", "-oi"]);
                assert_eq!(command.environment_policy, EnvironmentPolicy::Clear);
                assert!(command.working_directory.unwrap().is_absolute());
                assert_eq!(input, b"mail body");
                Ok(CompleteProcessOutput {
                    exit: ProcessExit {
                        code: Some(if success { 0 } else { 7 }),
                        signal: None,
                        success,
                    },
                    stdout: Vec::new(),
                    stderr: if success {
                        Vec::new()
                    } else {
                        b"failed".to_vec()
                    },
                })
            })
        }
    }

    fn context(calls: Rc<Cell<usize>>, success: bool) -> RuntimeContext {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_process(Rc::new(RecordingProcess { calls, success })),
            )
    }

    #[test]
    fn command_uses_one_installed_owned_input_call_and_preserves_exit() {
        let _serial = runmat_filesystem::provider_override_lock();
        let _native =
            runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
        let calls = Rc::new(Cell::new(0));
        let runtime = context(calls.clone(), true);
        futures::executor::block_on(run(
            Some(&runtime),
            PathBuf::from("/usr/sbin/sendmail"),
            None,
            &["a@example.test".into()],
            "mail body".into(),
        ))
        .expect("successful owned command");
        assert_eq!(calls.get(), 1);

        let runtime = context(calls.clone(), false);
        let error = futures::executor::block_on(run(
            Some(&runtime),
            PathBuf::from("/usr/sbin/sendmail"),
            None,
            &["a@example.test".into()],
            "mail body".into(),
        ))
        .expect_err("nonzero command exit");
        assert_eq!(
            error.identifier(),
            Some(SendmailFailure::Command.identifier())
        );
        assert!(error.message().contains("failed"));
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn unpolled_command_and_missing_service_have_no_process_effect() {
        let calls = Rc::new(Cell::new(0));
        let runtime = context(calls.clone(), true);
        drop(run(
            Some(&runtime),
            PathBuf::from("/usr/sbin/sendmail"),
            None,
            &[],
            "mail body".into(),
        ));
        assert_eq!(calls.get(), 0);
        let missing =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let error = futures::executor::block_on(run(
            Some(&missing),
            PathBuf::from("/usr/sbin/sendmail"),
            None,
            &[],
            "mail body".into(),
        ))
        .expect_err("missing process capability");
        assert_eq!(
            error.identifier(),
            Some("RunMat:RuntimeContext:CapabilityUnavailable")
        );
        assert_eq!(calls.get(), 0);
    }
}
