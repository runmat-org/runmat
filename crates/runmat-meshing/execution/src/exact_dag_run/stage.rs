use runmat_execution::value::ValuePayload;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_runner::AttemptSuccess;

use crate::{
    import_result_publication, import_stage_evidence_observation, MeshingProgressSink,
    MeshingStageKernel, PlannedMeshingStage,
};

use super::ExactMeshingDagRunError;

#[derive(Debug, thiserror::Error)]
pub enum MeshingStageExecutionError {
    #[error("{0}")]
    Message(String),
    #[error("{0}")]
    Failure(Box<runmat_meshing_core::MeshingFailure>),
}

impl MeshingStageExecutionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    pub const fn failure(&self) -> Option<&runmat_meshing_core::MeshingFailure> {
        match self {
            Self::Message(_) => None,
            Self::Failure(failure) => Some(failure),
        }
    }
}

pub trait ExactMeshingDagExecutor: CacheImport + CacheExport {
    fn execute_stage(
        &mut self,
        stage: &PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<AttemptSuccess, MeshingStageExecutionError>;

    /// Executes an algorithmically independent canonical partition batch. Implementations may
    /// schedule the batch concurrently but must return successes in the same order as `stages`.
    /// The default preserves the serial reference behavior.
    fn execute_stages(
        &mut self,
        stages: &[PlannedMeshingStage],
        revision: ProgramRevision,
    ) -> Result<Vec<AttemptSuccess>, MeshingStageExecutionError> {
        stages
            .iter()
            .map(|stage| self.execute_stage(stage, revision.clone()))
            .collect()
    }
}

pub struct SerialExactMeshingExecutor<'a, S> {
    pub store: &'a mut S,
    pub kernel: &'a dyn MeshingStageKernel,
    pub cancellation: &'a dyn runmat_meshing_core::MeshingCancellationSignal,
    pub progress: &'a mut dyn MeshingProgressSink,
    pub chunk_policy: runmat_meshing_core::MeshingChunkPolicy,
    pub inventory_limits: ObjectInventoryLimits,
}

impl<S: CacheImport> CacheImport for SerialExactMeshingExecutor<'_, S> {
    fn read_verified(
        &self,
        digest: runmat_execution::Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
}

impl<S: CacheExport> CacheExport for SerialExactMeshingExecutor<'_, S> {
    fn write_verified(
        &mut self,
        object: &runmat_execution_artifact::LogicalObject,
    ) -> runmat_execution_artifact::ArtifactResult<()> {
        self.store.write_verified(object)
    }
}

impl<S: CacheImport + CacheExport> ExactMeshingDagExecutor for SerialExactMeshingExecutor<'_, S> {
    fn execute_stage(
        &mut self,
        stage: &PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<AttemptSuccess, MeshingStageExecutionError> {
        crate::execute_serial_stage(
            &stage
                .program_request(revision)
                .map_err(|error| MeshingStageExecutionError::new(error.to_string()))?,
            self.store,
            self.kernel,
            self.cancellation,
            self.progress,
            self.chunk_policy,
            self.inventory_limits,
        )
        .map(|completed| completed.attempt_success())
        .map_err(map_serial_execution_error)
    }
}

fn map_serial_execution_error(
    error: crate::MeshingSerialExecutionError,
) -> MeshingStageExecutionError {
    match error {
        crate::MeshingSerialExecutionError::Stage(failure) => {
            MeshingStageExecutionError::Failure(failure)
        }
        crate::MeshingSerialExecutionError::Bridge(error) => {
            MeshingStageExecutionError::new(error.to_string())
        }
    }
}

pub(super) struct ExecutedStage {
    pub root: runmat_execution::value::ValueRef,
    pub result_objects: Vec<runmat_execution::value::ValueRef>,
    pub evidence: runmat_meshing_core::MeshingStageEvidence,
}

pub(super) fn execute_planned_stage<E: ExactMeshingDagExecutor>(
    executor: &mut E,
    stage: &PlannedMeshingStage,
    revision: ProgramRevision,
    limits: ObjectInventoryLimits,
) -> Result<ExecutedStage, ExactMeshingDagRunError> {
    let success = executor.execute_stage(stage, revision)?;
    admit_stage_success(executor, stage, success, limits)
}

pub(super) fn admit_stage_success<E: ExactMeshingDagExecutor>(
    executor: &E,
    stage: &PlannedMeshingStage,
    success: AttemptSuccess,
    limits: ObjectInventoryLimits,
) -> Result<ExecutedStage, ExactMeshingDagRunError> {
    let [ValuePayload::Object(root)] = success.outputs.as_slice() else {
        return Err(crate::MeshingExecutionError::Invalid(
            "meshing stage executor must return one externalized root".into(),
        )
        .into());
    };
    let root = root.as_ref().clone();
    let publication = import_result_publication(
        executor,
        &root,
        stage.host().artifact_access.clone(),
        limits,
    )?;
    if success.result_objects.len() != publication.result_objects().len() + 1
        || !publication
            .result_objects()
            .iter()
            .all(|reference| success.result_objects.contains(reference))
    {
        return Err(crate::MeshingExecutionError::Identity(
            "meshing stage result inventory contains missing or unrelated objects",
        )
        .into());
    }
    let evidence =
        import_stage_evidence_observation(executor, stage.host(), &success.result_objects, limits)?;
    Ok(ExecutedStage {
        root,
        result_objects: success.result_objects,
        evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_executor_preserves_typed_stage_failures() {
        let failure = runmat_meshing_core::MeshingFailure {
            schema_version: runmat_meshing_core::MESHING_FAILURE_SCHEMA_VERSION,
            category: runmat_meshing_core::MeshingFailureCategory::InvalidGeometry,
            stage: runmat_meshing_core::MeshingStageKind::SurfaceMesh,
            operation: runmat_meshing_core::MeshingOperation::TriangulateSurface,
            entity_ids: Vec::new(),
            witnesses: Vec::new(),
            request_values: Vec::new(),
            achieved_values: Vec::new(),
            remediation: "repair the exact boundary".into(),
        };

        let error = map_serial_execution_error(crate::MeshingSerialExecutionError::Stage(
            Box::new(failure.clone()),
        ));

        assert_eq!(error.failure(), Some(&failure));
    }
}
