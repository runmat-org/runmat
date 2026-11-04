//! Cancellation and time admission apply to cache IO as well as fresh workers.
use super::*;
use crate::NativeProgramSession;
use runmat_meshing_core::{MeshingFailure, MeshingFailureCategory, MESHING_FAILURE_SCHEMA_VERSION};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(super) struct CacheControl<'a> {
    session: &'a NativeProgramSession,
    started: Instant,
    maximum_wall: Duration,
    deadline: Option<u64>,
    stage: MeshingStageKind,
}
impl<'a> CacheControl<'a> {
    pub(super) fn new(
        session: &'a NativeProgramSession,
        stage: &PlannedMeshingStage,
        deadline: Option<u64>,
        started: Instant,
    ) -> Self {
        Self {
            session,
            started,
            maximum_wall: Duration::from_millis(stage.host().ceilings.maximum_wall_time_ms),
            deadline,
            stage: stage.host().workload.stage,
        }
    }
    pub(super) fn check(&self) -> Result<(), MeshingStageExecutionError> {
        let expired = self.started.elapsed() >= self.maximum_wall
            || self.deadline.is_some_and(|deadline| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(true, |elapsed| elapsed.as_millis() >= u128::from(deadline))
            });
        let category = if self.session.is_cancelled() {
            MeshingFailureCategory::Cancelled
        } else if expired {
            MeshingFailureCategory::TimeBudgetExceeded
        } else {
            return Ok(());
        };
        Err(MeshingStageExecutionError::Failure(Box::new(MeshingFailure {
            schema_version: MESHING_FAILURE_SCHEMA_VERSION, category, stage: self.stage, operation: self.stage.operation(),
            entity_ids: Vec::new(), witnesses: Vec::new(), request_values: Vec::new(), achieved_values: Vec::new(),
            accounting_error: None,
            remediation: "inspect original cancellation or time admission; no cached completion is accepted".into(),
        })))
    }
    fn artifact_check(&self) -> ArtifactResult<()> {
        self.check().map_err(|failure| {
            runmat_execution_artifact::ArtifactError::Invalid(failure.to_string())
        })
    }
}
pub(super) struct ControlledSource<'a, S> {
    pub store: &'a S,
    pub control: &'a CacheControl<'a>,
}
impl<S: CacheImport> CacheImport for ControlledSource<'_, S> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.control.artifact_check()?;
        let result = self.store.read_verified(digest)?;
        self.control.artifact_check()?;
        Ok(result)
    }
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> runmat_execution_artifact::ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.control.artifact_check()?;
        let result =
            self.store
                .read_verified_bounded_with_admission(digest, maximum_bytes, admit)?;
        self.control.artifact_check()?;
        Ok(result)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.control.artifact_check()?;
        let result = self.store.read_verified_bounded(digest, maximum_bytes)?;
        self.control.artifact_check()?;
        Ok(result)
    }
}
pub(super) struct ControlledDestination<'a, S> {
    pub store: &'a mut S,
    pub control: &'a CacheControl<'a>,
}
impl<S: CacheExport> CacheExport for ControlledDestination<'_, S> {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        self.control.artifact_check()?;
        self.store.write_verified(object)?;
        self.control.artifact_check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_cancellation_and_expired_envelopes_reject_before_cache_io() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
        config.store_root = directory.path().join("session");
        config.max_workers = 1;
        let session = NativeProgramSession::new(config).unwrap();
        let mut control = CacheControl {
            session: &session,
            started: Instant::now(),
            maximum_wall: Duration::ZERO,
            deadline: None,
            stage: MeshingStageKind::CurveMesh,
        };
        let failure = control.check().unwrap_err();
        assert_eq!(
            failure.failure().unwrap().category,
            MeshingFailureCategory::TimeBudgetExceeded
        );
        control.maximum_wall = Duration::from_secs(60);
        control.started = Instant::now() - Duration::from_secs(61);
        assert_eq!(
            control.check().unwrap_err().failure().unwrap().category,
            MeshingFailureCategory::TimeBudgetExceeded
        );
        control.started = Instant::now();
        control.deadline = Some(0);
        assert_eq!(
            control.check().unwrap_err().failure().unwrap().category,
            MeshingFailureCategory::TimeBudgetExceeded
        );
        control.deadline = None;
        assert!(control.check().is_ok());
        session.cancel(runmat_execution::CancellationReason::User);
        assert_eq!(
            control.check().unwrap_err().failure().unwrap().category,
            MeshingFailureCategory::Cancelled
        );
        struct Forbidden;
        impl CacheImport for Forbidden {
            fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
                panic!("cancelled cache must not read");
            }
        }
        let source = ControlledSource {
            store: &Forbidden,
            control: &control,
        };
        assert!(source.read_verified(Digest::sha256(b"cached")).is_err());
    }
}
