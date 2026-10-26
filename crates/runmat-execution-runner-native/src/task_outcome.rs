//! Native delivery classification from original typed authorities, never diagnostic text.
use crate::{NativeCheckpointEvidence, NativeExecutionError};
use runmat_execution::TaskId;
use runmat_execution_runner::{
    AttemptExecutionTermination, AttemptSuccess, TaskCancellation, TaskExecutionEvidence,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeTaskQuiescence {
    Confirmed,
    Unconfirmed,
}

#[derive(Clone, Debug)]
pub(crate) enum NativeTaskFailure {
    Checkpoint(NativeCheckpointEvidence),
    Cancelled {
        task_id: TaskId,
        cancellation: TaskCancellation,
        quiescence: NativeTaskQuiescence,
    },
    Scheduler(String),
}

impl NativeTaskFailure {
    pub(crate) fn from_scheduler(
        evidence: &TaskExecutionEvidence,
        result: &Result<AttemptSuccess, String>,
    ) -> Option<Self> {
        let Err(diagnostic) = result else {
            return None;
        };
        if evidence.task.state == runmat_execution::state::TaskState::Indeterminate {
            return Some(Self::Scheduler(diagnostic.clone()));
        }
        match evidence.task.cancellation {
            Some(cancellation) => Some(Self::Cancelled {
                task_id: evidence.task.id(),
                cancellation,
                quiescence: if evidence.attempts.iter().all(|attempt| {
                    attempt
                        .terminal
                        .as_ref()
                        .and_then(|terminal| terminal.execution.as_ref())
                        .is_some_and(|execution| {
                            matches!(
                                execution.termination,
                                AttemptExecutionTermination::NotStarted
                                    | AttemptExecutionTermination::ProcessExited { .. }
                            )
                        })
                }) {
                    NativeTaskQuiescence::Confirmed
                } else {
                    NativeTaskQuiescence::Unconfirmed
                },
            }),
            None => Some(Self::Scheduler(diagnostic.clone())),
        }
    }

    pub(crate) fn from_checkpoint(checkpoint: &NativeCheckpointEvidence) -> Option<Self> {
        match checkpoint {
            NativeCheckpointEvidence::Durable => None,
            NativeCheckpointEvidence::NotPublished { .. }
            | NativeCheckpointEvidence::DurabilityIndeterminate { .. } => {
                Some(Self::Checkpoint(checkpoint.clone()))
            }
        }
    }

    pub(crate) fn error(&self) -> NativeExecutionError {
        match self {
            Self::Checkpoint(evidence) => NativeExecutionError::CheckpointDelivery {
                diagnostic: evidence
                    .delivery_error()
                    .expect("failed checkpoint has an original diagnostic"),
                evidence: Box::new(evidence.clone()),
            },
            Self::Cancelled {
                task_id,
                cancellation,
                quiescence,
            } => NativeExecutionError::TaskCancelled {
                task_id: *task_id,
                cancellation: *cancellation,
                quiescence: *quiescence,
            },
            Self::Scheduler(message) => NativeExecutionError::Protocol(message.clone()),
        }
    }

    pub(crate) fn confirm_remote_quiescence(&mut self, confirmed: bool) {
        if let Self::Cancelled { quiescence, .. } = self {
            *quiescence = if confirmed {
                NativeTaskQuiescence::Confirmed
            } else {
                NativeTaskQuiescence::Unconfirmed
            };
        }
    }
}
