use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use runmat_analysis_execution::{
    import_published_result, import_structural_accuracy, import_structural_linear_completion,
    import_structural_observations, import_structural_recovered_fields, prepare_result_publication,
    prepare_structural_accuracy, prepare_structural_field_recovery, prepare_structural_observation,
    AnalysisHostWorkload, PreparedAnalysisProgram, PreparedStructuralLinearProgram,
    ResultPublicationPreparation, StructuralAccuracyPreparation,
    StructuralFieldRecoveryPreparation, StructuralObservationPreparation,
};
use runmat_analysis_fea::execution::{
    SequenceIdentity, StructuralLinearWorkload, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::AttemptSuccess;
use runmat_execution_runner_native::{
    FilesystemObjectStore, NativeProgramSession, RemotePoolDriver,
};

use super::{analysis_submission, wait_for_native_task};

pub(super) trait PublicationChainDriver {
    type Store: CacheImport;

    fn store(&self) -> &Self::Store;

    fn execute<'a>(
        &'a mut self,
        host: &'a AnalysisHostWorkload,
        program: &'a PreparedAnalysisProgram,
    ) -> Pin<Box<dyn Future<Output = AttemptSuccess> + 'a>>;
}

pub(super) async fn execute_publication_chain<D: PublicationChainDriver>(
    driver: &mut D,
    solve: &PreparedStructuralLinearProgram,
    solve_success: &AttemptSuccess,
    program_revision: ProgramRevision,
) -> PublishedAnalysisResult {
    let limits = ObjectInventoryLimits::default();
    let imported_solve = import_structural_linear_completion(
        driver.store(),
        success_response(solve_success),
        solve,
        limits,
    )
    .unwrap();
    let source_result = imported_solve.objects;
    let source_numerical_receipt = imported_solve.numerical_receipt_objects;
    let workload =
        StructuralLinearWorkload::from_canonical_bytes(&solve.inputs[0].payload()).unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap();
    let step = workload.procedure_step().unwrap().clone();

    let fields = prepare_structural_field_recovery(StructuralFieldRecoveryPreparation {
        source_host: solve.host.clone(),
        source_result: source_result.clone(),
        mesh: solve.inputs[1].clone(),
        program_revision: program_revision.clone(),
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: limits,
    })
    .unwrap();
    let field_success = driver.execute(&fields.host, &fields.program).await;
    let imported_fields = import_structural_recovered_fields(
        driver.store(),
        success_response(&field_success),
        &fields,
        limits,
    )
    .unwrap();
    let observations = prepare_structural_observation(
        driver.store(),
        StructuralObservationPreparation {
            source_host: fields.host.clone(),
            source_request: fields.request.clone(),
            source_fields: imported_fields,
            mesh: solve.inputs[1].clone(),
            sequence,
            step: step.clone(),
            limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
            program_revision: program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: limits,
        },
    )
    .unwrap();
    let observation_success = driver
        .execute(&observations.host, &observations.program)
        .await;
    let imported_observations = import_structural_observations(
        driver.store(),
        success_response(&observation_success),
        &observations,
        limits,
    )
    .unwrap();

    let accuracy = prepare_structural_accuracy(StructuralAccuracyPreparation {
        solve_host: solve.host.clone(),
        source_numerical_receipt,
        source_observations: Some(imported_observations.clone()),
        sequence,
        step,
        formulation_requirement: workload.formulation_requirement,
        program_revision: program_revision.clone(),
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: limits,
    })
    .unwrap();
    let accuracy_success = driver.execute(&accuracy.host, &accuracy.program).await;
    let imported_accuracy = import_structural_accuracy(
        driver.store(),
        success_response(&accuracy_success),
        &accuracy,
        limits,
    )
    .unwrap();

    let publication = prepare_result_publication(ResultPublicationPreparation {
        accuracy_host: accuracy.host.clone(),
        source_accuracy: imported_accuracy.objects,
        source_observations: Some(imported_observations),
        program_revision,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: limits,
    })
    .unwrap();
    let publication_success = driver
        .execute(&publication.host, &publication.program)
        .await;
    import_published_result(
        driver.store(),
        success_response(&publication_success),
        &publication,
        limits,
    )
    .unwrap()
    .result
}

pub(super) fn success_response(success: &AttemptSuccess) -> ProgramExecutionResponse {
    ProgramExecutionResponse::ExternalizedSuccess {
        outputs: success.outputs.clone(),
        result_objects: success.result_objects.clone(),
    }
}

pub(super) fn response_to_success(response: ProgramExecutionResponse) -> AttemptSuccess {
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        panic!("analysis stage did not externalize its result: {response:?}")
    };
    AttemptSuccess {
        outputs,
        result_objects,
    }
}

pub(super) struct ProcessDriver<'a> {
    pub session: &'a NativeProgramSession,
    pub store: &'a mut FilesystemObjectStore,
}

impl PublicationChainDriver for ProcessDriver<'_> {
    type Store = FilesystemObjectStore;

    fn store(&self) -> &Self::Store {
        self.store
    }

    fn execute<'a>(
        &'a mut self,
        host: &'a AnalysisHostWorkload,
        program: &'a PreparedAnalysisProgram,
    ) -> Pin<Box<dyn Future<Output = AttemptSuccess> + 'a>> {
        Box::pin(async move {
            program.write_to(self.store).unwrap();
            let task = self
                .session
                .submit_graph(vec![(
                    program.request.clone(),
                    analysis_submission(
                        self.session.scope_id(),
                        self.session.pool_id(),
                        host,
                        &program.request,
                    ),
                )])
                .unwrap()
                .pop()
                .unwrap();
            wait_for_native_task(task).await
        })
    }
}

pub(super) struct DurableDriver {
    config: runmat_execution_runner_native::supervisor::LocalSupervisorConfig,
    supervisor: Arc<runmat_execution_runner_native::supervisor::LocalSupervisor>,
    store: FilesystemObjectStore,
    submission_sequence: u64,
}

impl DurableDriver {
    pub fn open(config: runmat_execution_runner_native::supervisor::LocalSupervisorConfig) -> Self {
        let supervisor =
            runmat_execution_runner_native::supervisor::LocalSupervisor::open(config.clone())
                .unwrap();
        let store = supervisor.object_store().unwrap();
        Self {
            config,
            supervisor,
            store,
            submission_sequence: 0,
        }
    }
}

impl PublicationChainDriver for DurableDriver {
    type Store = FilesystemObjectStore;

    fn store(&self) -> &Self::Store {
        &self.store
    }

    fn execute<'a>(
        &'a mut self,
        _host: &'a AnalysisHostWorkload,
        program: &'a PreparedAnalysisProgram,
    ) -> Pin<Box<dyn Future<Output = AttemptSuccess> + 'a>> {
        Box::pin(async move {
            program.write_to(&mut self.store).unwrap();
            self.submission_sequence += 1;
            let idempotency_key =
                format!("analysis-publication-chain-{}", self.submission_sequence);
            let (record, duplicate) = self
                .supervisor
                .submit_program(
                    runmat_execution_runner_native::supervisor::ProgramBatchSubmission::from_request(
                        program.request.clone(),
                        Some(idempotency_key),
                        runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
                    ),
                )
                .await
                .unwrap();
            assert!(!duplicate);
            let response = tokio::time::timeout(Duration::from_secs(15), async {
                loop {
                    self.supervisor.reconcile().await.unwrap();
                    let attachment = self
                        .supervisor
                        .attach(record.handle.id, 0, 0)
                        .await
                        .unwrap();
                    if attachment.record.state.is_terminal() {
                        break attachment.response.unwrap_or_else(|| {
                            panic!(
                                "durable analysis failed without response: {:?}",
                                attachment.record.message
                            )
                        });
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .expect("durable analysis completion timeout");
            let success = response_to_success(response);

            self.supervisor = runmat_execution_runner_native::supervisor::LocalSupervisor::open(
                self.config.clone(),
            )
            .unwrap();
            self.store = self.supervisor.object_store().unwrap();
            success
        })
    }
}

#[derive(Default)]
pub(super) struct MemoryStore(pub BTreeMap<Digest, Vec<u8>>);

impl CacheImport for MemoryStore {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

pub(super) struct RemoteDriver {
    pub pool: Arc<RemotePoolDriver>,
    pub store: MemoryStore,
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
}

impl PublicationChainDriver for RemoteDriver {
    type Store = MemoryStore;

    fn store(&self) -> &Self::Store {
        &self.store
    }

    fn execute<'a>(
        &'a mut self,
        host: &'a AnalysisHostWorkload,
        program: &'a PreparedAnalysisProgram,
    ) -> Pin<Box<dyn Future<Output = AttemptSuccess> + 'a>> {
        Box::pin(async move {
            for object in &program.objects {
                self.pool
                    .register_execution_object(object.reference.clone(), object.bytes.clone())
                    .unwrap();
            }
            let completion = self
                .pool
                .submit_graph(vec![(
                    program.request.clone(),
                    analysis_submission(self.scope_id, self.pool_id, host, &program.request),
                )])
                .unwrap()
                .pop()
                .unwrap();
            let success =
                match tokio::time::timeout(Duration::from_secs(15), completion.wait()).await {
                    Ok(result) => result.unwrap(),
                    Err(_) => panic!(
                        "remote publication-chain completion timeout: {:?}",
                        self.pool.snapshot()
                    ),
                };
            for reference in &success.result_objects {
                self.store.0.insert(
                    reference.logical_digest,
                    self.pool
                        .execution_object(reference)
                        .unwrap()
                        .expect("verified remote publication-chain result")
                        .to_vec(),
                );
            }
            success
        })
    }
}
