use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use runmat_analysis_execution::{
    extend_structural_resource_usage, import_accuracy, import_observation_assembly,
    import_observation_partition_contribution, import_published_result, import_reduced_observation,
    import_structural_linear_completion, import_structural_recovered_fields, prepare_accuracy,
    prepare_observation_assembly, prepare_observation_fanout, prepare_observation_reduction,
    prepare_result_publication, prepare_structural_field_recovery, AccuracyPreparation,
    AnalysisHostWorkload, ObservationAssemblyPreparation, ObservationPreparation,
    ObservationReductionPreparation, ObservationWorkloadSource, PlannedResourceUsageGraph,
    PreparedAnalysisProgram, PreparedResourceUsage, PreparedStructuralLinearProgram,
    ResultPublicationPreparation, StructuralFieldRecoveryPreparation,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, SequenceIdentity, StructuralLinearWorkload,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::{AttemptResourceAccounting, AttemptSuccess};
use runmat_execution_runner_native::{
    FilesystemObjectStore, NativeProgramSession, RemotePoolDriver,
};
use runmat_meshing_core::CanonicalMeshingContract;

use super::{analysis_resources, analysis_submission, capabilities, wait_for_native_task};

#[path = "publication_chain/diagnostics.rs"]
mod diagnostics;
use diagnostics::remote_timeout_summary;

pub(super) trait PublicationChainDriver {
    type Store: CacheImport;

    fn store(&self) -> &Self::Store;

    fn execute<'a>(
        &'a mut self,
        host: &'a AnalysisHostWorkload,
        program: &'a PreparedAnalysisProgram,
        resource_plan: Option<&'a runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    ) -> Pin<Box<dyn Future<Output = MeteredProgramResponse> + 'a>>;

    fn execute_batch<'a>(
        &'a mut self,
        workloads: Vec<(
            &'a AnalysisHostWorkload,
            &'a PreparedAnalysisProgram,
            Option<&'a runmat_analysis_fea::planning::resource::FeaResourcePlan>,
        )>,
    ) -> Pin<Box<dyn Future<Output = Vec<MeteredProgramResponse>> + 'a>> {
        Box::pin(async move {
            let mut responses = Vec::with_capacity(workloads.len());
            for (host, program, resource_plan) in workloads {
                responses.push(self.execute(host, program, resource_plan).await);
            }
            responses
        })
    }
}

pub(super) async fn execute_publication_chain<D: PublicationChainDriver>(
    driver: &mut D,
    solve: &PreparedStructuralLinearProgram,
    solve_response: ProgramExecutionResponse,
    resource_usage: &PreparedResourceUsage,
    program_revision: ProgramRevision,
) -> PublishedAnalysisResult {
    let limits = ObjectInventoryLimits::default();
    let imported_solve =
        import_structural_linear_completion(driver.store(), solve_response, solve, limits).unwrap();
    let source_result = imported_solve.objects.clone();
    let source_numerical_receipt = imported_solve.numerical_receipt_objects.clone();
    let source_invocation = solve
        .inputs
        .iter()
        .find(|input| {
            input.manifest.kind
                == runmat_analysis_fea::execution::AnalysisNumericalObjectKind::SolverInvocation
        })
        .unwrap()
        .clone();
    let mesh = solve
        .inputs
        .iter()
        .find(|input| {
            input.manifest.kind == runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh
        })
        .cloned();
    let accelerator_requests = resource_usage
        .evidence
        .plans
        .iter()
        .find(|plan| plan.identity == resource_usage.evidence.primary_plan)
        .expect("resource evidence must contain its primary plan")
        .request
        .accelerators
        .clone();
    let workload =
        StructuralLinearWorkload::from_canonical_bytes(&source_invocation.payload()).unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap();
    let step = workload.procedure_step().unwrap().clone();

    let fields = prepare_structural_field_recovery(StructuralFieldRecoveryPreparation {
        source_host: solve.host.clone(),
        source_invocation: source_invocation.clone(),
        source_result: source_result.clone(),
        mesh: mesh.clone(),
        program_revision: program_revision.clone(),
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: limits,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
    })
    .unwrap();
    let field_response = driver
        .execute(&fields.host, &fields.program, Some(&fields.resource_plan))
        .await;
    let imported_fields = import_structural_recovered_fields(
        driver.store(),
        field_response.response,
        &fields,
        limits,
    )
    .unwrap();
    let observation_graph_started = Instant::now();
    let field_catalog =
        runmat_analysis_fea::results::observation::ObservationFieldCatalog::structural(
            fields.request.clone(),
            imported_fields.manifest.clone(),
        )
        .unwrap();
    let fanout = prepare_observation_fanout(
        driver.store(),
        ObservationPreparation {
            source_host: fields.host.clone(),
            source: runmat_analysis_execution::ObservationPublishedSource::Structural {
                recovery_request: fields.request.clone(),
                fields: imported_fields,
            },
            source_workload: ObservationWorkloadSource::Linear(source_invocation.clone()),
            mesh: mesh.clone(),
            accelerator_requests,
            sequence,
            step: step.clone(),
            limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
            program_revision: program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: limits,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
        runmat_analysis_fea::results::field::RecoveredFieldSample::Final,
        None,
    )
    .unwrap();
    let responses = driver
        .execute_batch(
            fanout
                .partitions
                .iter()
                .map(|partition| {
                    (
                        &partition.host,
                        &partition.program,
                        Some(&partition.resource_plan),
                    )
                })
                .collect(),
        )
        .await;
    let mut observation_attempts = Vec::new();
    let mut contribution_groups = BTreeMap::new();
    for (partition, response) in fanout.partitions.iter().zip(responses) {
        let execution = response
            .phase_execution()
            .expect("complete original observation task history");
        execution
            .validate_for(
                &partition.host,
                &partition.resource_plan,
                &response.response,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "original observation validation failed: {error}; {}",
                    diagnostics::attempt_summary(
                        response.execution.as_ref().expect("original task history")
                    ),
                )
            });
        observation_attempts.extend(
            execution
                .resource_graph(&partition.resource_plan)
                .unwrap_or_else(|error| {
                    panic!(
                        "original observation history is not fully accounted: {error}; {}",
                        diagnostics::attempt_summary(
                            response.execution.as_ref().expect("original task history")
                        ),
                    )
                })
                .attempts,
        );
        let imported = import_observation_partition_contribution(
            driver.store(),
            response.response,
            partition,
            limits,
        )
        .unwrap();
        contribution_groups
            .entry(partition.partition_plan.identity)
            .or_insert_with(|| (partition.partition_plan.clone(), Vec::new()))
            .1
            .push(imported);
    }
    let mut reduced = Vec::new();
    for (_, (partition_plan, contributions)) in contribution_groups {
        let reduction = prepare_observation_reduction(ObservationReductionPreparation {
            source_host: fields.host.clone(),
            source_request: fanout.source_request.clone(),
            partition_plan,
            contributions,
            program_revision: program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: limits,
        })
        .unwrap();
        let response = driver
            .execute(
                &reduction.host,
                &reduction.program,
                Some(&reduction.resource_plan),
            )
            .await;
        let execution = response
            .phase_execution()
            .expect("complete original observation task history");
        execution
            .validate_for(
                &reduction.host,
                &reduction.resource_plan,
                &response.response,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "original observation validation failed: {error}; {}",
                    diagnostics::attempt_summary(
                        response.execution.as_ref().expect("original task history")
                    ),
                )
            });
        observation_attempts.extend(
            execution
                .resource_graph(&reduction.resource_plan)
                .unwrap()
                .attempts,
        );
        reduced.push(
            import_reduced_observation(driver.store(), response.response, &reduction, limits)
                .unwrap(),
        );
    }
    let partition_evidence: BTreeMap<_, _> = reduced
        .iter()
        .map(|imported| {
            (
                imported.result.observation_id.clone(),
                imported.result.partition_evidence.clone(),
            )
        })
        .collect();
    let observations = prepare_observation_assembly(ObservationAssemblyPreparation {
        source_host: fields.host.clone(),
        source_request: fanout.source_request,
        results: reduced,
        program_revision: program_revision.clone(),
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: limits,
    })
    .unwrap();
    let observation_response = driver
        .execute(
            &observations.host,
            &observations.program,
            Some(&observations.resource_plan),
        )
        .await;
    let execution = observation_response
        .phase_execution()
        .expect("complete original observation task history");
    execution
        .validate_for(
            &observations.host,
            &observations.resource_plan,
            &observation_response.response,
        )
        .unwrap();
    observation_attempts.extend(
        execution
            .resource_graph(&observations.resource_plan)
            .unwrap()
            .attempts,
    );
    let imported_observations = import_observation_assembly(
        driver.store(),
        observation_response.response,
        &observations,
        limits,
    )
    .unwrap();
    let observation_resources = PlannedResourceUsageGraph::new(
        observation_attempts,
        u64::try_from(observation_graph_started.elapsed().as_millis())
            .unwrap_or(u64::MAX)
            .max(1),
    )
    .unwrap();
    let decoded_mesh = mesh.as_ref().map(|objects| {
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&objects.payload()).unwrap()
    });
    let publication_catalog =
        runmat_analysis_fea::results::observation::ObservationPublicationCatalog::new(
            solve.host.workload.step,
            observations.source_request.plan.clone(),
            field_catalog,
            &step,
            vec![
                runmat_analysis_fea::results::observation::ObservationPublicationEntry::Sample(
                    Box::new(
                        runmat_analysis_fea::results::observation::ObservationSamplePublication {
                            request: observations.source_request.clone(),
                            manifest: imported_observations.manifest.clone(),
                            partition_evidence: imported_observations
                                .manifest
                                .results
                                .iter()
                                .map(|entry| partition_evidence[&entry.observation_id].clone())
                                .collect(),
                        },
                    ),
                ),
            ],
            runmat_analysis_fea::execution::ObservationHistorySourceContext::Structural {
                source: runmat_analysis_fea::execution::StructuralObservationSource::Linear(
                    &workload,
                ),
                mesh: decoded_mesh.as_ref(),
            },
        )
        .unwrap();
    let expected_attempt_count =
        resource_usage.evidence.attempts.len() + observation_resources.attempts.len();
    let resource_usage = extend_structural_resource_usage(
        solve,
        &imported_solve,
        resource_usage,
        &observation_resources,
        DEFAULT_ANALYSIS_CHUNK_BYTES,
        limits,
    )
    .unwrap();
    assert_eq!(
        resource_usage.evidence.attempts.len(),
        expected_attempt_count
    );

    let accuracy = prepare_accuracy(AccuracyPreparation {
        solve_host: solve.host.clone(),
        source_kind: runmat_analysis_fea::execution::AccuracySourceKind::StructuralLinear,
        source_numerical_receipt,
        source_observations: Some(publication_catalog.clone()),
        source_context_inputs: solve
            .inputs
            .iter()
            .filter(|input| {
                matches!(
                    input.manifest.kind,
                    AnalysisNumericalObjectKind::SolverInvocation
                        | AnalysisNumericalObjectKind::Mesh
                )
            })
            .cloned()
            .collect(),
        sequence,
        step,
        program_revision: program_revision.clone(),
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: limits,
    })
    .unwrap();
    let accuracy_response = driver
        .execute(
            &accuracy.host,
            &accuracy.program,
            Some(&accuracy.resource_plan),
        )
        .await;
    let accuracy_execution = accuracy_response
        .phase_execution()
        .expect("complete original accuracy task history");
    accuracy_execution
        .validate_for(
            &accuracy.host,
            &accuracy.resource_plan,
            &accuracy_response.response,
        )
        .unwrap();
    let accuracy_graph = accuracy_execution
        .resource_graph(&accuracy.resource_plan)
        .unwrap();
    let imported_accuracy = import_accuracy(
        driver.store(),
        accuracy_response.response,
        &accuracy,
        limits,
    )
    .unwrap();

    let resource_usage = runmat_analysis_execution::extend_resource_usage(
        &resource_usage,
        &accuracy_graph,
        &solve.host.artifact_access,
        runmat_analysis_execution::AnalysisOutputPolicy::new(DEFAULT_ANALYSIS_CHUNK_BYTES, limits)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        resource_usage.evidence.attempts.len(),
        expected_attempt_count + 1
    );

    let publication = prepare_result_publication(ResultPublicationPreparation {
        accuracy_host: accuracy.host.clone(),
        source_accuracy: imported_accuracy.objects,
        source_resource_usage: resource_usage.objects.clone(),
        source_observations: Some(publication_catalog),
        program_revision,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: limits,
    })
    .unwrap();
    let publication_response = driver
        .execute(
            &publication.host,
            &publication.program,
            Some(&publication.resource_plan),
        )
        .await;
    let publication_execution = publication_response
        .phase_execution()
        .expect("complete original publication history");
    let published = import_published_result(
        driver.store(),
        publication_response.response,
        &publication,
        limits,
    )
    .unwrap();
    let terminal = runmat_analysis_execution::complete_result_publication_resource_usage(
        &publication,
        &published,
        &resource_usage,
        &publication_execution,
        runmat_analysis_execution::AnalysisOutputPolicy::new(DEFAULT_ANALYSIS_CHUNK_BYTES, limits)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(terminal.evidence.attempts.len(), expected_attempt_count + 2);
    assert_eq!(
        published.result.source_resource_usage,
        resource_usage.evidence.identity()
    );
    runmat_analysis_execution::validate_retained_result_publication_resource_usage(
        &published.result,
        &resource_usage.evidence,
        &terminal.evidence,
    )
    .unwrap();
    published.result
}

#[derive(Clone)]
pub(super) struct MeteredProgramResponse {
    pub response: ProgramExecutionResponse,
    pub resources: AttemptResourceAccounting,
    pub execution: Option<runmat_execution_runner::TaskExecutionEvidence>,
    pub graph_wall_millis: u64,
}

impl MeteredProgramResponse {
    pub fn phase_execution(
        &self,
    ) -> Result<runmat_analysis_execution::AnalysisPhaseExecutionEvidence, String> {
        let original = self.execution.clone().ok_or_else(|| {
            "durable supervisor completion lacks the original portable admitted task/history"
                .to_string()
        })?;
        let phase = runmat_analysis_execution::AnalysisPhaseExecutionEvidence::new(
            original,
            self.graph_wall_millis,
        )
        .map_err(|error| error.to_string())?;
        if phase.winning_accounting() != &self.resources {
            return Err("response accounting differs from original portable commit".into());
        }
        Ok(phase)
    }
}

pub(super) fn success_response(success: &AttemptSuccess) -> ProgramExecutionResponse {
    ProgramExecutionResponse::from_success_outputs(
        success.values().expect("analysis value result").0.to_vec(),
        success.result_objects().to_vec(),
    )
}

pub(super) fn metered_response(
    success: AttemptSuccess,
    execution: runmat_execution_runner::TaskExecutionEvidence,
    graph_wall_millis: u64,
) -> MeteredProgramResponse {
    success.validate_accounting().unwrap();
    let resources = success
        .resources()
        .expect("analysis task must retain resource accounting")
        .clone();
    let response = success_response(&success);
    MeteredProgramResponse {
        response,
        resources,
        execution: Some(execution),
        graph_wall_millis,
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
        resource_plan: Option<&'a runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    ) -> Pin<Box<dyn Future<Output = MeteredProgramResponse> + 'a>> {
        Box::pin(async move {
            let started = Instant::now();
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
                        resource_plan,
                        resource_plan
                            .map(|plan| plan.request.clone())
                            .unwrap_or_else(analysis_resources),
                    ),
                )])
                .unwrap()
                .pop()
                .unwrap();
            {
                let success = wait_for_native_task(&task).await;
                metered_response(
                    success,
                    task.try_execution_evidence()
                        .expect("original native terminal history"),
                    u64::try_from(started.elapsed().as_millis().max(1)).unwrap(),
                )
            }
        })
    }

    fn execute_batch<'a>(
        &'a mut self,
        workloads: Vec<(
            &'a AnalysisHostWorkload,
            &'a PreparedAnalysisProgram,
            Option<&'a runmat_analysis_fea::planning::resource::FeaResourcePlan>,
        )>,
    ) -> Pin<Box<dyn Future<Output = Vec<MeteredProgramResponse>> + 'a>> {
        Box::pin(async move {
            let started = Instant::now();
            for (_, program, _) in &workloads {
                program.write_to(self.store).unwrap();
            }
            let tasks = self
                .session
                .submit_graph(
                    workloads
                        .iter()
                        .map(|(host, program, resource_plan)| {
                            (
                                program.request.clone(),
                                analysis_submission(
                                    self.session.scope_id(),
                                    self.session.pool_id(),
                                    host,
                                    &program.request,
                                    *resource_plan,
                                    resource_plan
                                        .map(|plan| plan.request.clone())
                                        .unwrap_or_else(analysis_resources),
                                ),
                            )
                        })
                        .collect(),
                )
                .unwrap();
            let mut responses = Vec::with_capacity(tasks.len());
            for task in tasks {
                responses.push({
                    let success = wait_for_native_task(&task).await;
                    metered_response(
                        success,
                        task.try_execution_evidence()
                            .expect("original native terminal history"),
                        u64::try_from(started.elapsed().as_millis().max(1)).unwrap(),
                    )
                });
            }
            let wall = u64::try_from(started.elapsed().as_millis().max(1)).unwrap();
            for response in &mut responses {
                response.graph_wall_millis = wall;
            }
            responses
        })
    }
}

pub(super) struct DurableDriver {
    config: runmat_execution_runner_native::supervisor::LocalSupervisorConfig,
    supervisor: Option<Arc<runmat_execution_runner_native::supervisor::LocalSupervisor>>,
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
            supervisor: Some(supervisor),
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
        host: &'a AnalysisHostWorkload,
        program: &'a PreparedAnalysisProgram,
        resource_plan: Option<&'a runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    ) -> Pin<Box<dyn Future<Output = MeteredProgramResponse> + 'a>> {
        Box::pin(async move {
            let started = Instant::now();
            program.write_to(&mut self.store).unwrap();
            // The durable supervisor accepts generic programs, so this driver must carry
            // the same host requirements that analysis task submission normally adds.
            let mut allocation = resource_plan
                .map(|plan| plan.request.clone())
                .unwrap_or_else(analysis_resources);
            allocation.required_capabilities.extend(capabilities());
            allocation
                .required_capabilities
                .extend(host.workload.generic_capabilities().unwrap());
            self.submission_sequence += 1;
            let idempotency_key =
                format!("analysis-publication-chain-{}", self.submission_sequence);
            let context = self
                .supervisor
                .as_ref()
                .unwrap()
                .execution_context(Some(&idempotency_key))
                .await
                .unwrap();
            let original_task = analysis_submission(
                context.scope_id,
                context.pool_id,
                host,
                &program.request,
                resource_plan,
                allocation.clone(),
            );
            let (record, duplicate) = self
                .supervisor
                .as_ref()
                .expect("durable supervisor is open")
                .submit_program(
                    runmat_execution_runner_native::supervisor::ProgramBatchSubmission::from_request(
                        program.request.clone(),
                        original_task.clone(),
                        Some(idempotency_key),
                        runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
                    ),
                )
                .await
                .unwrap();
            assert!(!duplicate);
            let response = tokio::time::timeout(Duration::from_secs(15), async {
                loop {
                    let supervisor = self
                        .supervisor
                        .as_ref()
                        .expect("durable supervisor is open");
                    supervisor.reconcile().await.unwrap();
                    let attachment = supervisor.attach(record.handle.id, 0, 0).await.unwrap();
                    if attachment.record.state.is_terminal() {
                        let resources = attachment.resources.as_ref().expect(
                            "durable analysis task must retain generic resource accounting",
                        );
                        resources.validate().unwrap();
                        assert_eq!(resources.resources.allocation, allocation);
                        let response = attachment.response.unwrap_or_else(|| {
                            panic!(
                                "durable analysis failed without response: {:?}",
                                attachment.record.message
                            )
                        });
                        break MeteredProgramResponse {
                            response,
                            resources: resources.clone(),
                            execution: attachment.execution,
                            graph_wall_millis: u64::try_from(started.elapsed().as_millis().max(1))
                                .unwrap(),
                        };
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .expect("durable analysis completion timeout");
            // Reattach under a new exclusive owner after releasing the previous
            // supervisor. The object store does not own the authority lease.
            drop(self.supervisor.take());
            let supervisor = runmat_execution_runner_native::supervisor::LocalSupervisor::open(
                self.config.clone(),
            )
            .unwrap();
            self.store = supervisor.object_store().unwrap();
            self.supervisor = Some(supervisor);
            response
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

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        match self.0.get(&digest) {
            Some(bytes) if bytes.len() as u64 > maximum_bytes => {
                Err(runmat_execution_artifact::ArtifactError::Limit(
                    "object exceeds its read bound".into(),
                ))
            }
            bytes => Ok(bytes.cloned()),
        }
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
        resource_plan: Option<&'a runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    ) -> Pin<Box<dyn Future<Output = MeteredProgramResponse> + 'a>> {
        Box::pin(async move {
            let started = Instant::now();
            if let Some(plan) = resource_plan {
                let inventory = super::inventory();
                assert!(
                    plan.request.memory_bytes <= inventory.memory_bytes
                        && plan.request.scratch_bytes <= inventory.scratch_bytes,
                    "remote conformance worker cannot admit {:?}: request {:?}, inventory {:?}",
                    host.workload.kind,
                    plan.request,
                    inventory,
                );
            }
            for (reference, bytes) in program.objects() {
                self.pool
                    .register_execution_object(reference.clone(), bytes.to_vec())
                    .unwrap();
            }
            self.pool
                .register_program_input_inventory(
                    &program.request,
                    program
                        .objects()
                        .map(|(reference, _)| reference.clone())
                        .collect(),
                )
                .unwrap();
            let submission = analysis_submission(
                self.scope_id,
                self.pool_id,
                host,
                &program.request,
                resource_plan,
                resource_plan
                    .map(|plan| plan.request.clone())
                    .unwrap_or_else(analysis_resources),
            );
            let task_id = submission.request.id;
            let completion = self
                .pool
                .submit_graph(vec![(program.request.clone(), submission)])
                .unwrap()
                .pop()
                .unwrap();
            let success =
                match tokio::time::timeout(Duration::from_secs(15), completion.wait()).await {
                    Ok(result) => result.unwrap(),
                    Err(_) => panic!(
                        "remote publication-chain completion timeout: {:?}",
                        remote_timeout_summary(&self.pool)
                    ),
                };
            for reference in success.result_objects() {
                self.store.0.insert(
                    reference.logical_digest,
                    self.pool
                        .execution_object(reference)
                        .unwrap()
                        .expect("verified remote publication-chain result")
                        .to_vec(),
                );
            }
            let evidence = self.pool.task_execution_evidence(task_id).unwrap();
            metered_response(
                success,
                evidence,
                u64::try_from(started.elapsed().as_millis().max(1)).unwrap(),
            )
        })
    }

    fn execute_batch<'a>(
        &'a mut self,
        workloads: Vec<(
            &'a AnalysisHostWorkload,
            &'a PreparedAnalysisProgram,
            Option<&'a runmat_analysis_fea::planning::resource::FeaResourcePlan>,
        )>,
    ) -> Pin<Box<dyn Future<Output = Vec<MeteredProgramResponse>> + 'a>> {
        Box::pin(async move {
            let started = Instant::now();
            for (_, program, _) in &workloads {
                for (reference, bytes) in program.objects() {
                    self.pool
                        .register_execution_object(reference.clone(), bytes.to_vec())
                        .unwrap();
                }
                self.pool
                    .register_program_input_inventory(
                        &program.request,
                        program
                            .objects()
                            .map(|(reference, _)| reference.clone())
                            .collect(),
                    )
                    .unwrap();
            }
            let submissions = workloads
                .iter()
                .map(|(host, program, resource_plan)| {
                    (
                        program.request.clone(),
                        analysis_submission(
                            self.scope_id,
                            self.pool_id,
                            host,
                            &program.request,
                            *resource_plan,
                            resource_plan
                                .map(|plan| plan.request.clone())
                                .unwrap_or_else(analysis_resources),
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let task_ids = submissions
                .iter()
                .map(|(_, submission)| submission.request.id)
                .collect::<Vec<_>>();
            let completions = self.pool.submit_graph(submissions).unwrap();
            let mut responses = Vec::with_capacity(completions.len());
            for (completion, task_id) in completions.into_iter().zip(task_ids) {
                let success =
                    match tokio::time::timeout(Duration::from_secs(15), completion.wait()).await {
                        Ok(result) => result.unwrap(),
                        Err(_) => panic!(
                            "remote publication-chain batch timeout: {:?}",
                            remote_timeout_summary(&self.pool)
                        ),
                    };
                for reference in success.result_objects() {
                    self.store.0.insert(
                        reference.logical_digest,
                        self.pool
                            .execution_object(reference)
                            .unwrap()
                            .expect("verified remote publication-chain batch result")
                            .to_vec(),
                    );
                }
                let evidence = self.pool.task_execution_evidence(task_id).unwrap();
                responses.push(metered_response(success, evidence, 1));
            }
            let wall = u64::try_from(started.elapsed().as_millis().max(1)).unwrap();
            for response in &mut responses {
                response.graph_wall_millis = wall;
            }
            responses
        })
    }
}
