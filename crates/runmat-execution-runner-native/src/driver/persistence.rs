//! Serialize snapshot persistence with its scheduler state, and retain uncertainty.
use super::LocalDriver;
use crate::{NativeCheckpointEvidence, NativeExecutionResult};
use runmat_execution_runner::DriverSnapshot;

impl LocalDriver {
    /// Caller holds the driver mutex from snapshot capture through publication.
    pub(super) fn persist_snapshot(
        &self,
        snapshot: &DriverSnapshot,
    ) -> NativeExecutionResult<NativeCheckpointEvidence> {
        #[cfg(test)]
        if self
            .fail_next_checkpoint
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return Err(crate::NativeExecutionError::Protocol(
                "injected checkpoint publication failure".into(),
            ));
        }
        let publication = self.checkpoints.write(snapshot);
        let evidence = match &publication {
            Ok(outcome) => NativeCheckpointEvidence::from_publication(outcome.clone()),
            Err(error) => NativeCheckpointEvidence::NotPublished {
                diagnostic: error.to_string(),
            },
        };
        *self
            .checkpoint_evidence
            .lock()
            .expect("checkpoint evidence poisoned") = evidence.clone();
        if let Some(diagnostic) = evidence.delivery_error() {
            tracing::error!(driver_fence = snapshot.driver_fence, %diagnostic, "native checkpoint persistence uncertain");
        }
        publication.map(|_| evidence)
    }

    pub(super) fn checkpoint(&self) -> NativeExecutionResult<NativeCheckpointEvidence> {
        let driver = self.driver.lock().expect("local driver poisoned");
        self.persist_snapshot(&driver.snapshot())
    }

    pub(super) fn persist_observed_snapshot(
        &self,
        snapshot: &DriverSnapshot,
    ) -> NativeCheckpointEvidence {
        match self.persist_snapshot(snapshot) {
            Ok(evidence) => evidence,
            Err(error) => NativeCheckpointEvidence::NotPublished {
                diagnostic: error.to_string(),
            },
        }
    }

    pub(super) fn checkpoint_observed(&self) -> NativeCheckpointEvidence {
        let driver = self.driver.lock().expect("local driver poisoned");
        self.persist_observed_snapshot(&driver.snapshot())
    }

    pub(crate) fn checkpoint_evidence(&self) -> NativeCheckpointEvidence {
        self.checkpoint_evidence
            .lock()
            .expect("checkpoint evidence poisoned")
            .clone()
    }
}
