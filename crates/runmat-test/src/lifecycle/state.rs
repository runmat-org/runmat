use crate::result::{ResultState, TerminalDisposition};

use super::QualificationKind;
use crate::executor::ExecutionFault;

#[derive(Clone, Debug)]
pub(crate) struct LifecycleState {
    pub result: ResultState,
    pub abort_test: bool,
    pub abort_run: bool,
}

impl Default for LifecycleState {
    fn default() -> Self {
        Self {
            result: ResultState::PASSED,
            abort_test: false,
            abort_run: false,
        }
    }
}

impl LifecycleState {
    pub fn apply_qualification(&mut self, kind: QualificationKind) {
        use QualificationKind::*;
        match kind {
            VerificationFailed => {
                self.result.failed = true;
                self.strengthen(TerminalDisposition::Failed);
            }
            AssumptionFailed => {
                self.result.incomplete = true;
                self.strengthen(TerminalDisposition::Filtered);
                self.abort_test = true;
            }
            AssertionFailed => {
                self.result.failed = true;
                self.result.incomplete = true;
                self.strengthen(TerminalDisposition::Failed);
                self.abort_test = true;
            }
            FatalAssertionFailed => {
                self.result.failed = true;
                self.result.incomplete = true;
                self.strengthen(TerminalDisposition::Failed);
                self.abort_test = true;
                self.abort_run = true;
            }
        }
    }

    pub fn apply_fault(&mut self, fault: &ExecutionFault) {
        self.result.incomplete = true;
        self.abort_test = true;
        let (failed, disposition) = match fault {
            ExecutionFault::Uncaught(_) => (true, TerminalDisposition::Failed),
            ExecutionFault::TimedOut(_) => (true, TerminalDisposition::TimedOut),
            ExecutionFault::Cancelled(_) => (false, TerminalDisposition::Cancelled),
            ExecutionFault::WorkerCrashed(_) => (true, TerminalDisposition::Crashed),
        };
        self.result.failed |= failed;
        self.strengthen(disposition);
    }

    fn strengthen(&mut self, disposition: TerminalDisposition) {
        if disposition_rank(disposition) > disposition_rank(self.result.disposition) {
            self.result.disposition = disposition;
        }
    }
}

fn disposition_rank(disposition: TerminalDisposition) -> u8 {
    match disposition {
        TerminalDisposition::Passed => 0,
        TerminalDisposition::Filtered => 1,
        TerminalDisposition::Failed => 2,
        TerminalDisposition::Cancelled => 3,
        TerminalDisposition::TimedOut => 4,
        TerminalDisposition::Crashed => 5,
    }
}
