//! One serial analysis worker using the shared scheduler and accounting contracts.
use crate::*;
use runmat_execution::{
    host::ExecutionHostInventory,
    identity::{ArtifactId, WorkerId},
    resource::ResourceInventory,
    state::PoolState,
    task::RetryPolicy,
    value::ValuePayload,
    Digest, ExecutionScopeId, PoolId,
};
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    ProgramExecutionResponse,
};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand, DriverConfig, PoolSpec,
    WorkerSpec,
};
use std::collections::{BTreeMap, BTreeSet};

pub struct SerialAnalysisHost {
    driver: Driver,
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    resources: ResourceInventory,
}

#[derive(Clone, Copy)]
pub struct SerialAnalysisServices<'a> {
    /// Host clock used for attempt accounting.
    pub clock: &'a dyn Fn() -> u64,
    /// Admission owner for additional numerical work.
    pub capacity: Option<&'a dyn AsyncCapacityOwner>,
}
impl SerialAnalysisHost {
    pub fn new(
        host: ExecutionHostInventory,
        resources: ResourceInventory,
        nonce: Digest,
    ) -> AnalysisBridgeResult<Self> {
        if resources.cpu_millicores != 1000 || !resources.accelerators.is_empty() {
            return Err(AnalysisBridgeError::Invalid(
                "Serial execution requires exactly one CPU core and no accelerators.".into(),
            ));
        }
        let scope_id = ExecutionScopeId::derive(&[b"serial-analysis", nonce.bytes()]);
        let pool_id = PoolId::derive(&[scope_id.bytes(), b"serial"]);
        let worker_id = WorkerId::derive(&[pool_id.bytes(), b"worker"]);
        let mut driver = Driver::new(DriverConfig::default(), 1).map_err(error)?;
        driver
            .handle(DriverCommand::RegisterScope {
                scope_id,
                parent: None,
            })
            .map_err(error)?;
        driver
            .handle(DriverCommand::CreatePool(PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 1,
                resource_limit: resources.clone(),
            }))
            .map_err(error)?;
        driver
            .handle(DriverCommand::SetPoolState {
                pool_id,
                state: PoolState::Ready,
            })
            .map_err(error)?;
        driver
            .handle(DriverCommand::RegisterWorker(WorkerSpec {
                id: worker_id,
                pool_id,
                resources: resources.clone(),
                host,
            }))
            .map_err(error)?;
        Ok(Self {
            driver,
            scope_id,
            pool_id,
            resources,
        })
    }
    pub fn execute<S: CacheImport + CacheExport>(
        &mut self,
        host: &AnalysisHostWorkload,
        program: &PreparedAnalysisProgram,
        inputs: &[PreparedAnalysisObjects],
        plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
        store: &mut S,
        clock: &dyn Fn() -> u64,
    ) -> AnalysisBridgeResult<(ProgramExecutionResponse, AttemptSuccess)> {
        crate::serial::finish_inline(self.execute_with_capacity(
            host,
            program,
            inputs,
            plan,
            store,
            SerialAnalysisServices {
                clock,
                capacity: None,
            },
        ))
    }

    pub async fn execute_with_capacity<S: CacheImport + CacheExport>(
        &mut self,
        host: &AnalysisHostWorkload,
        program: &PreparedAnalysisProgram,
        inputs: &[PreparedAnalysisObjects],
        plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
        store: &mut S,
        services: SerialAnalysisServices<'_>,
    ) -> AnalysisBridgeResult<(ProgramExecutionResponse, AttemptSuccess)> {
        let SerialAnalysisServices { clock, capacity } = services;
        plan.validate_identity()?;
        // Use the same workload capability binding as native submission. The
        // resource plan owns numeric admission; task placement also requires
        // the actual analysis host ABI and any formulation capabilities.
        let mut resources = plan.request.clone();
        resources
            .required_capabilities
            .extend(host.workload.generic_capabilities()?);
        if plan.request.cpu_millicores > self.resources.cpu_millicores
            || plan.request.memory_bytes > self.resources.memory_bytes
            || plan.request.scratch_bytes > self.resources.scratch_bytes
            || !plan.request.accelerators.is_empty()
            || !resources
                .required_capabilities
                .is_subset(&self.resources.capabilities)
        {
            return Err(AnalysisBridgeError::Invalid(
                "The prepared analysis exceeds this execution host's admitted resources.".into(),
            ));
        }
        if !matches!(
            host.workload.kind,
            runmat_analysis_fea::execution::AnalysisWorkKind::Solve
                | runmat_analysis_fea::execution::AnalysisWorkKind::PublishState
        ) {
            return Err(AnalysisBridgeError::Invalid(
                "This procedure requires another execution host.".into(),
            ));
        }
        let roots = program
            .request
            .arguments
            .iter()
            .map(|value| match value {
                ValuePayload::Object(root) => Ok((**root).clone()),
                _ => Err(AnalysisBridgeError::Invalid(
                    "Prepared analysis input is not externalized.".into(),
                )),
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let requirement = program
            .request
            .artifact
            .execution_host_requirement(
                &program.request.recipe,
                BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
            )
            .map_err(error)?;
        let submission = build_resource_planned_task_submission_with_closure(
            host,
            plan,
            PreparedAnalysisTaskClosure {
                inputs,
                output_policy: host.output_policy,
            },
            roots,
            Vec::new(),
            BTreeSet::new(),
            AnalysisExecutionContext {
                scope_id: self.scope_id,
                pool_id: self.pool_id,
                program_artifact_id: ArtifactId::derive(&[program.request.artifact.id.0.bytes()]),
                host: requirement,
                resources,
                retry: RetryPolicy::Never,
                deadline_unix_millis: None,
                priority: 0,
            },
        )?;
        program.write_to(store)?;
        let actions = self
            .driver
            .handle(DriverCommand::Submit(Box::new(submission)))
            .map_err(error)?;
        let attempt = actions
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(attempt) => Some(attempt),
                _ => None,
            })
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "No compatible serial worker admitted this analysis.".into(),
                )
            })?;
        self.driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &attempt,
                AttemptReport::Started,
            )))
            .map_err(error)?;
        let started = clock();
        let completed = if let Some(capacity) = capacity {
            execute_serial_analysis_host_response_with_capacity(
                &program.request,
                store,
                &SerialStructuralAnalysisKernel,
                host.output_policy.chunk_bytes(),
                host.output_policy.inventory_limits(),
                Some(&attempt.task.resources),
                capacity,
            )
            .await?
        } else {
            execute_serial_analysis_host_response(
                &program.request,
                store,
                &SerialStructuralAnalysisKernel,
                host.output_policy.chunk_bytes(),
                host.output_policy.inventory_limits(),
                Some(&attempt.task.resources),
            )?
        };
        let response = completed.program_response();
        let ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } = &response
        else {
            return Err(AnalysisBridgeError::Invalid(
                "Serial analysis did not externalize its results.".into(),
            ));
        };
        let success = AttemptSuccess::new(
            &attempt,
            outputs.clone(),
            result_objects.clone(),
            clock().saturating_sub(started).max(1),
            0,
            0,
        )
        .map_err(error)?;
        self.driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &attempt,
                AttemptReport::Succeeded {
                    result: success.clone(),
                },
            )))
            .map_err(error)?;
        Ok((response, success))
    }
}
fn error(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}

pub fn execute_serial_structural_sequence<S: CacheImport + CacheExport>(
    preparation: StructuralSequencePreparation,
    store: &mut S,
    host: &mut SerialAnalysisHost,
    clock: &dyn Fn() -> u64,
    cancelled: &dyn Fn() -> bool,
) -> AnalysisBridgeResult<CompletedStructuralSequence> {
    crate::serial::finish_inline(execute_serial_structural_sequence_with_capacity(
        preparation,
        store,
        host,
        clock,
        cancelled,
        None,
    ))
}

pub async fn execute_serial_structural_sequence_with_capacity<S: CacheImport + CacheExport>(
    preparation: StructuralSequencePreparation,
    store: &mut S,
    host: &mut SerialAnalysisHost,
    clock: &dyn Fn() -> u64,
    cancelled: &dyn Fn() -> bool,
    capacity: Option<&dyn AsyncCapacityOwner>,
) -> AnalysisBridgeResult<CompletedStructuralSequence> {
    let mut coordinator = StructuralSequenceCoordinator::new(preparation)?;
    while !coordinator.is_complete() {
        if cancelled() {
            return Err(AnalysisBridgeError::Invalid("Simulation stopped.".into()));
        }
        let solve_wave = coordinator.prepare_next_solve_wave()?;
        let mut responses = BTreeMap::new();
        let mut attempts = BTreeMap::new();
        for task in &solve_wave.tasks {
            let (response, attempt) = host
                .execute_with_capacity(
                    &task.prepared.host,
                    &task.prepared.program,
                    &task.prepared.inputs,
                    &task.prepared.resource_plan,
                    store,
                    SerialAnalysisServices { clock, capacity },
                )
                .await?;
            responses.insert(task.node.clone(), response);
            attempts.insert(task.node.clone(), attempt);
        }
        let state_wave =
            coordinator.complete_metered_solve_wave(store, solve_wave, responses, attempts)?;
        let mut responses = BTreeMap::new();
        let mut attempts = BTreeMap::new();
        for task in &state_wave.tasks {
            if cancelled() {
                return Err(AnalysisBridgeError::Invalid("Simulation stopped.".into()));
            }
            let (response, success) = host
                .execute_with_capacity(
                    &task.prepared.host,
                    &task.prepared.program,
                    &task.prepared.inputs,
                    &task.prepared.resource_plan,
                    store,
                    SerialAnalysisServices { clock, capacity },
                )
                .await?;
            let accounting = success.resources().cloned().ok_or_else(|| {
                AnalysisBridgeError::Invalid("Serial task omitted its accounting.".into())
            })?;
            responses.insert(task.node.clone(), response);
            attempts.insert(
                task.node.clone(),
                PlannedResourceUsageAttempt {
                    plan: task.prepared.resource_plan.clone(),
                    accounting,
                },
            );
        }
        coordinator.complete_metered_state_wave(store, state_wave, responses, attempts)?;
    }
    coordinator.finish()
}
