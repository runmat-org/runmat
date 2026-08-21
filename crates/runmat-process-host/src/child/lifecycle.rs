use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};

use super::CapturedStderr;
use crate::{ProcessHostError, ProcessHostResult};

pub struct ChildStdio {
    pub stdin: ChildStdin,
    pub stdout: ChildStdout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessExit {
    pub code: Option<i32>,
    pub success: bool,
}

pub struct ChildProcess {
    child: Child,
    process_id: Option<u32>,
    stderr: CapturedStderr,
    stdio: Option<ChildStdio>,
    _containment: Option<super::process_tree::ProcessContainment>,
}

impl ChildProcess {
    pub(super) fn new(
        child: Child,
        process_id: Option<u32>,
        stderr: CapturedStderr,
        containment: Option<super::process_tree::ProcessContainment>,
    ) -> Self {
        Self {
            child,
            process_id,
            stderr,
            stdio: None,
            _containment: containment,
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

    pub fn id(&self) -> Option<u32> {
        self.process_id
    }

    pub fn captured_stderr(&self) -> CapturedStderr {
        self.stderr.clone()
    }

    pub async fn wait(&mut self) -> ProcessHostResult<ProcessExit> {
        let status = self.child.wait().await?;
        Ok(ProcessExit {
            code: status.code(),
            success: status.success(),
        })
    }

    pub fn try_wait(&mut self) -> ProcessHostResult<Option<ProcessExit>> {
        Ok(self.child.try_wait()?.map(|status| ProcessExit {
            code: status.code(),
            success: status.success(),
        }))
    }

    pub async fn terminate_tree(&mut self) -> ProcessHostResult<()> {
        super::process_tree::terminate(&mut self.child, self.process_id).await?;
        Ok(())
    }
}
