//! Atomic native completion; original execution and checkpoint delivery stay separate.
use crate::{NativeCheckpointEvidence, NativeExecutionResult};
use runmat_execution_runner::{AttemptSuccess, TaskExecutionEvidence};

pub struct NativeProgramCompletion {
    result: Result<AttemptSuccess, String>,
    evidence: Option<TaskExecutionEvidence>,
    checkpoint: NativeCheckpointEvidence,
}

impl NativeProgramCompletion {
    pub(crate) fn new(
        result: Result<AttemptSuccess, String>,
        evidence: Option<TaskExecutionEvidence>,
        checkpoint: NativeCheckpointEvidence,
    ) -> Self {
        Self {
            result,
            evidence,
            checkpoint,
        }
    }
    pub fn scheduler_result(&self) -> &Result<AttemptSuccess, String> {
        &self.result
    }
    pub fn execution_evidence(&self) -> Option<&TaskExecutionEvidence> {
        self.evidence.as_ref()
    }
    pub fn checkpoint_evidence(&self) -> &NativeCheckpointEvidence {
        &self.checkpoint
    }

    /// Durable delivery takes priority over a scheduler cancellation or success.
    pub fn delivery_result(&self) -> NativeExecutionResult<&AttemptSuccess> {
        if let Some(failure) =
            crate::task_outcome::NativeTaskFailure::from_checkpoint(&self.checkpoint)
        {
            return Err(failure.error());
        }
        let evidence = self.evidence.as_ref().ok_or_else(|| {
            crate::NativeExecutionError::Protocol(
                "native completion has no original scheduler history".into(),
            )
        })?;
        evidence.validate()?;
        if let Some(failure) =
            crate::task_outcome::NativeTaskFailure::from_scheduler(evidence, &self.result)
        {
            return Err(failure.error());
        }
        if let Ok(result) = &self.result {
            if evidence
                .task
                .committed
                .as_ref()
                .is_none_or(|commit| commit.result != *result)
            {
                return Err(crate::NativeExecutionError::Protocol(
                    "native delivery differs from its original commit".into(),
                ));
            }
        }
        self.result
            .as_ref()
            .map_err(|message| crate::NativeExecutionError::Protocol(message.clone()))
    }
    /// Transfer the validated original output without cloning it again. Borrow
    /// `delivery_result` instead when the caller also needs to retain evidence.
    pub fn into_delivery_result(self) -> NativeExecutionResult<AttemptSuccess> {
        self.delivery_result()?;
        self.result.map_err(crate::NativeExecutionError::Protocol)
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Result<AttemptSuccess, String>,
        Option<TaskExecutionEvidence>,
        NativeCheckpointEvidence,
    ) {
        (self.result, self.evidence, self.checkpoint)
    }
}
