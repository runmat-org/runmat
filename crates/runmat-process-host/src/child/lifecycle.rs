use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};

use super::CapturedStderr;
use crate::command::ProcessGroupPlacement;
use crate::{ProcessHostError, ProcessHostResult};

pub struct ChildStdio {
    pub stdin: ChildStdin,
    pub stdout: ChildStdout,
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
    stderr: CapturedStderr,
    stdio: Option<ChildStdio>,
    _containment: Option<super::process_tree::ProcessContainment>,
    process_group_placement: ProcessGroupPlacement,
}

impl ChildProcess {
    pub(super) fn new(
        child: Child,
        process_id: Option<u32>,
        stderr: CapturedStderr,
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

    pub fn id(&self) -> Option<u32> {
        self.process_id
    }

    /// A joined Unix member is not an independent process-tree certificate.
    pub fn owns_process_tree(&self) -> bool {
        self.process_group_placement == ProcessGroupPlacement::OwnGroup
    }

    pub fn captured_stderr(&self) -> CapturedStderr {
        self.stderr.clone()
    }

    pub async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        let status = self.child.wait().await?;
        Ok(process_exit(status))
    }

    pub fn try_wait(&mut self) -> ProcessHostResult<Option<ProcessExit>> {
        Ok(self.child.try_wait()?.map(process_exit))
    }

    pub async fn terminate_tree(&mut self) -> ProcessHostResult<()> {
        super::process_tree::terminate(
            &mut self.child,
            self.process_id,
            self._containment.as_ref(),
            self.process_group_placement,
        )
        .await?;
        Ok(())
    }

    /// Bounded exceptional-Drop retirement for an owned child whose async
    /// bootstrap future was cancelled. Joined children prove only member
    /// retirement; their ancestor remains responsible for its process group.
    pub(super) fn terminate_tree_blocking(&mut self) -> ProcessHostResult<()> {
        super::process_tree::terminate_blocking(
            &mut self.child,
            self.process_id,
            self._containment.as_ref(),
            self.process_group_placement,
        )?;
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
