use super::*;
use crate::analysis_task_pool::{AdmittedAnalysisBatch, AnalysisTaskCompletion};
use async_trait::async_trait;
use runmat_analysis_execution::{PreparedAnalysisObjects, PreparedAnalysisProgram};
use runmat_execution::{identity::WorkerId, resource::ResourceInventory, state::PoolState};
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport, FilesystemObjectStore},
    ArtifactAccess, ArtifactResult, ProgramExecutionObject, ProgramExecutionResponse,
};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand, DriverConfig, PoolSpec,
    TaskSubmission, WorkerSpec,
};

pub(crate) struct RecordingPool {
    pub(crate) store: FilesystemObjectStore,
    pub tasks: Vec<TaskSubmission>,
    driver: Option<Driver>,
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
            driver: None,
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
    async fn begin_wave_observation(
        &mut self,
        _: &[runmat_execution::TaskId],
    ) -> NativeExecutionResult<()> {
        Ok(())
    }

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
        tasks: AdmittedAnalysisBatch,
    ) -> NativeExecutionResult<Vec<AnalysisTaskCompletion>> {
        let mut results = Vec::new();
        for (program, submission) in tasks.into_submissions() {
            for dependency in &submission.dependencies {
                assert!(
                    self.tasks.iter().any(|task| task.request.id == *dependency),
                    "dependency was never executed on this pool"
                );
            }
            if self.driver.is_none() {
                self.driver = Some(recording_driver(&submission)?);
            }
            let driver = self.driver.as_mut().unwrap();
            let attempt = driver
                .handle(DriverCommand::Submit(Box::new(submission.clone())))
                .map_err(protocol)?
                .into_iter()
                .find_map(|action| match action {
                    DriverAction::Launch(request) => Some(request),
                    _ => None,
                })
                .ok_or_else(|| protocol("recording task was not admitted for execution"))?;
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
            let success = AttemptSuccess::new(
                &attempt,
                outputs.clone(),
                result_objects.clone(),
                1,
                bytes,
                0,
            )
            .map_err(protocol)?;
            driver
                .handle(DriverCommand::Tick {
                    now_millis: driver.now_millis() + 1,
                })
                .map_err(protocol)?;
            driver
                .handle(DriverCommand::BackendReport(BackendReport::for_request(
                    &attempt,
                    AttemptReport::Succeeded {
                        result: success.clone(),
                    },
                )))
                .map_err(protocol)?;
            let evidence = driver
                .snapshot()
                .task_execution_evidence(submission.request.id)
                .map_err(protocol)?;
            self.tasks.push(submission);
            results.push(AnalysisTaskCompletion::new(evidence, Ok(success))?);
        }
        Ok(results)
    }
}

// This test substrate executes kernels directly; the portable driver owns all
// admission/history/commit identities. Its accounting remains explicit test data.
fn recording_driver(submission: &TaskSubmission) -> NativeExecutionResult<Driver> {
    let scope = submission.request.scope_id;
    let pool = submission.request.pool_id;
    let inventory = ResourceInventory {
        cpu_millicores: 1000,
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 1024 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: submission.request.resources.required_capabilities.clone(),
    };
    let mut driver = Driver::new(DriverConfig::default(), 1).map_err(protocol)?;
    driver
        .handle(DriverCommand::RegisterScope {
            scope_id: scope,
            parent: None,
        })
        .map_err(protocol)?;
    driver
        .handle(DriverCommand::CreatePool(PoolSpec {
            id: pool,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory.clone(),
        }))
        .map_err(protocol)?;
    driver
        .handle(DriverCommand::SetPoolState {
            pool_id: pool,
            state: PoolState::Ready,
        })
        .map_err(protocol)?;
    driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: WorkerId::derive(&[b"recording-thermal-worker"]),
            pool_id: pool,
            resources: inventory,
        }))
        .map_err(protocol)?;
    Ok(driver)
}
