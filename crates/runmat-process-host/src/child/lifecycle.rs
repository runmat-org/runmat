use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};

use super::CapturedProcessOutput;
use crate::command::ProcessGroupPlacement;
use crate::{ProcessHostError, ProcessHostResult};

pub struct ChildStdio {
    pub stdin: ChildStdin,
    pub stdout: ChildStdout,
}

/// Complete application streams, without the private protocol transport.
pub struct ChildOutput {
    pub stdout: ChildStdout,
    pub stderr: ChildStderr,
    pub stderr_diagnostics: super::ProcessOutputCaptureWriter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessExit {
    pub code: Option<i32>,
    /// Unix signal termination is distinct from an ordinary exit code.
    pub signal: Option<i32>,
    pub success: bool,
}

pub struct ChildProcess {
    child: Child,
    process_id: Option<u32>,
    stderr: CapturedProcessOutput,
    stdio: Option<ChildStdio>,
    _containment: Option<super::process_tree::ProcessContainment>,
    process_group_placement: ProcessGroupPlacement,
    retirement: RetirementPhase,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RetirementPhase {
    Live,
    ExitObserved,
    Signaled,
    Retired,
}

impl ChildProcess {
    pub(super) fn new(
        child: Child,
        process_id: Option<u32>,
        stderr: CapturedProcessOutput,
        containment: Option<super::process_tree::ProcessContainment>,
        process_group_placement: ProcessGroupPlacement,
    ) -> Self {
        Self {
            child,
            process_id,
            stderr,
            stdio: None,
            _containment: containment,
            process_group_placement,
            retirement: RetirementPhase::Live,
        }
    }

    pub(super) fn install_stdio(&mut self, stdio: ChildStdio) {
        self.stdio = Some(stdio);
    }

    pub(super) fn child_stdin(&mut self) -> Option<ChildStdin> {
        self.child.stdin.take()
    }

    pub(super) fn child_stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }

    pub(super) fn child_stderr(&mut self) -> Option<ChildStderr> {
        self.child.stderr.take()
    }

    pub fn take_stdio(&mut self) -> ProcessHostResult<ChildStdio> {
        self.stdio
            .take()
            .ok_or_else(|| ProcessHostError::Protocol("child stdio has already been taken".into()))
    }

    pub(crate) fn take_complete_output(
        &mut self,
    ) -> ProcessHostResult<(Option<ChildStdin>, ChildStdout, ChildStderr)> {
        if self.stdio.is_some() || self.child.stdout.is_none() || self.child.stderr.is_none() {
            return Err(ProcessHostError::Protocol(
                "complete child output pipes are unavailable".into(),
            ));
        }
        Ok((
            self.child.stdin.take(),
            self.child.stdout.take().expect("checked stdout"),
            self.child.stderr.take().expect("checked stderr"),
        ))
    }

    pub fn take_application_output(&mut self) -> ProcessHostResult<ChildOutput> {
        let stderr_diagnostics = self.stderr.begin_capture()?;
        let (input, stdout, stderr) = self.take_complete_output()?;
        drop(input);
        Ok(ChildOutput {
            stdout,
            stderr,
            stderr_diagnostics,
        })
    }

    pub fn id(&self) -> Option<u32> {
        self.process_id
    }

    /// A joined Unix member is not an independent process-tree certificate.
    pub fn owns_process_tree(&self) -> bool {
        self.process_group_placement == ProcessGroupPlacement::OwnGroup
    }

    pub fn captured_stderr(&self) -> CapturedProcessOutput {
        self.stderr.clone()
    }

    pub async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        let status = self.child.wait().await?;
        Ok(process_exit(status))
    }

    /// Preserve the direct child's Unix PID until its owned group is signaled
    /// and retired. Call this instead of reaping and subsequently signaling a
    /// numeric group identifier that the OS could already have reused.
    pub async fn wait_and_retire_tree(&mut self) -> ProcessHostResult<ProcessExit> {
        #[cfg(unix)]
        if self.retirement == RetirementPhase::Live {
            super::exit_observation::wait_unreaped(self.process_id)
                .await
                .map_err(ProcessHostError::ExitObservation)?;
            self.retirement = RetirementPhase::ExitObserved;
        }
        #[cfg(not(unix))]
        let _ = self.child.wait().await?;
        self.terminate_tree().await?;
        self.wait().await
    }

    pub fn try_wait(&mut self) -> ProcessHostResult<Option<ProcessExit>> {
        Ok(self.child.try_wait()?.map(process_exit))
    }

    /// Observe leader exit while retaining the OS process-group owner for
    /// subsequent authenticated tree retirement.
    pub fn try_wait_unreaped(&mut self) -> ProcessHostResult<Option<ProcessExit>> {
        #[cfg(unix)]
        if matches!(
            self.retirement,
            RetirementPhase::Live | RetirementPhase::ExitObserved
        ) {
            let exit = super::exit_observation::try_wait_unreaped(self.process_id)
                .map_err(ProcessHostError::ExitObservation)?;
            if exit.is_some() {
                self.retirement = RetirementPhase::ExitObserved;
            }
            return Ok(exit);
        }
        self.try_wait()
    }

    pub async fn terminate_tree(&mut self) -> ProcessHostResult<()> {
        if self.retirement == RetirementPhase::Retired {
            return Ok(());
        }
        self.begin_retirement()?;
        super::process_tree::await_retirement(
            &mut self.child,
            self.process_id,
            self._containment.as_ref(),
            self.process_group_placement,
        )
        .await?;
        self.retirement = RetirementPhase::Retired;
        Ok(())
    }

    /// Bounded exceptional-Drop retirement for an owned child whose async
    /// bootstrap future was cancelled. Joined children prove only member
    /// retirement; their ancestor remains responsible for its process group.
    pub(super) fn terminate_tree_blocking(&mut self) -> ProcessHostResult<()> {
        if self.retirement == RetirementPhase::Retired {
            return Ok(());
        }
        self.begin_retirement()?;
        super::process_tree::await_retirement_blocking(
            &mut self.child,
            self.process_id,
            self._containment.as_ref(),
            self.process_group_placement,
        )?;
        self.retirement = RetirementPhase::Retired;
        Ok(())
    }

    fn begin_retirement(&mut self) -> ProcessHostResult<()> {
        if matches!(
            self.retirement,
            RetirementPhase::Live | RetirementPhase::ExitObserved
        ) {
            super::process_tree::begin_termination(
                &mut self.child,
                self.process_id,
                self._containment.as_ref(),
                self.process_group_placement,
                self.retirement == RetirementPhase::ExitObserved,
            )?;
            // Record synchronously, before the first await or possible reap.
            // Cancellation retries may finish waiting, never signal again.
            self.retirement = RetirementPhase::Signaled;
        }
        Ok(())
    }
}

fn process_exit(status: std::process::ExitStatus) -> ProcessExit {
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt as _;
        status.signal()
    };
    #[cfg(not(unix))]
    let signal = None;
    ProcessExit {
        code: status.code(),
        signal,
        success: status.success(),
    }
}
