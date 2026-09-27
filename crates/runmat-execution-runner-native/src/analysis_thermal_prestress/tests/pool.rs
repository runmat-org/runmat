use super::*;
use async_trait::async_trait;
use runmat_analysis_execution::{PreparedAnalysisObjects, PreparedAnalysisProgram};
use runmat_execution::{
    identity::{AttemptId, WorkerId},
    resource::ResourceAccounting,
};
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    ArtifactAccess, ArtifactResult, ProgramExecutionObject, ProgramExecutionRequest,
    ProgramExecutionResponse,
};
use runmat_execution_runner::{AttemptResourceAccounting, AttemptSuccess, TaskSubmission};

pub(crate) struct RecordingPool {
    pub(crate) store: FilesystemObjectStore,
    pub tasks: Vec<TaskSubmission>,
}
impl RecordingPool {
    pub fn new(path: &std::path::Path) -> Self {
        Self {
            store: FilesystemObjectStore::open(
                path,
                crate::NativeAnalysisHostLimits::default()
                    .inventory
                    .max_object_bytes,
            )
            .unwrap(),
            tasks: Vec::new(),
        }
    }
}
impl CacheImport for RecordingPool {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
}
#[async_trait]
impl AnalysisTaskPool for RecordingPool {
    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"recording-thermal-composition"])
    }
    fn pool_id(&self) -> PoolId {
        PoolId::derive(&[b"recording-thermal-composition"])
    }
    fn register_program(&mut self, program: &PreparedAnalysisProgram) -> NativeExecutionResult<()> {
        program.write_to(&mut self.store).map_err(protocol)
    }
    fn register_result_objects(
        &mut self,
        objects: &PreparedAnalysisObjects,
        _: &ArtifactAccess,
    ) -> NativeExecutionResult<()> {
        for object in &objects.objects {
            self.store.write_verified(object).map_err(protocol)?;
        }
        Ok(())
    }
    fn register_recovery_object(
        &mut self,
        _: &ProgramExecutionObject,
    ) -> NativeExecutionResult<()> {
        Err(protocol(
            "recording thermal test does not admit recovery objects",
        ))
    }
    async fn execute_batch(
        &mut self,
        tasks: Vec<(ProgramExecutionRequest, TaskSubmission)>,
        _: u64,
    ) -> NativeExecutionResult<Vec<(ProgramExecutionResponse, Option<AttemptSuccess>)>> {
        let mut results = Vec::new();
        for (program, submission) in tasks {
            for dependency in &submission.dependencies {
                assert!(
                    self.tasks.iter().any(|task| task.request.id == *dependency),
                    "dependency was never executed on this pool"
                );
            }
            let response = crate::execute_analysis_program_request(
                &program,
                &mut self.store,
                &crate::NativeAnalysisKernel,
                crate::NativeAnalysisHostLimits::default(),
            );
            let ProgramExecutionResponse::ExternalizedSuccess {
                outputs,
                result_objects,
            } = &response
            else {
                return Err(protocol(format!("recording worker failed: {response:?}")));
            };
            let bytes = result_objects
                .iter()
                .map(|object| object.encoded_length)
                .sum();
            let resources = AttemptResourceAccounting::new(
                AttemptId::derive(&[
                    b"recording-thermal-attempt",
                    &(self.tasks.len() as u64).to_le_bytes(),
                ]),
                WorkerId::derive(&[b"test-not-scheduler-evidence"]),
                ResourceAccounting::for_reserved_interval(
                    submission.request.resources.clone(),
                    1,
                    bytes,
                    0,
                    0,
                )
                .unwrap(),
            )
            .unwrap();
            let success = AttemptSuccess {
                outputs: outputs.clone(),
                result_objects: result_objects.clone(),
                resources: Some(Box::new(resources)),
            };
            self.tasks.push(submission);
            results.push((response, Some(success)));
        }
        Ok(results)
    }
}
