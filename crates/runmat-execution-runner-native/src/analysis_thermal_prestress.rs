//! Same-pool thermal lifecycle composition with the existing structural prestress owners.
use std::sync::Arc;

use runmat_analysis_core::StateSource;
use runmat_analysis_execution::{
    AnalysisOutputPolicy, CompletedThermalStudy, StructuralPrestressedFrequencyStudyPreparation,
    ThermalPrestressSource, ThermalStudyPreparation, ThermalStudyScope,
};
use runmat_analysis_fea::{
    execution::{AnalysisTaskIdentity, StructuralStepWorkload, StructuralStudyWorkloads},
    study::{CompiledAnalysisStudy, ThermalStudyProduct},
};
use runmat_execution::{ExecutionScopeId, PoolId};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, protocol, AnalysisTaskDependency, AnalysisTaskPool,
        LocalAnalysisTaskPool, RemoteAnalysisTaskPool,
    },
    CompletedStructuralBucklingSequenceStudy, CompletedStructuralPrestressedFrequencyStudy,
    CompletedStructuralPrestressedModalFrequencyStudy, CompletedStructuralPrestressedModalStudy,
    NativeExecutionConfig, NativeExecutionResult, NativeProgramSession, RemotePoolDriver,
    StructuralSequenceExecutionPolicy, ThermalStudyExecutionPolicy,
};

mod dispatch;
#[cfg(test)]
pub(crate) mod tests;

/// One original compiled definition and one shared execution context for every phase.
/// The structural preparation supplies the full workload inventory, mesh and admitted policies;
/// its common fields are reused by modal, buckling and both frequency branches.
pub struct ThermalPrestressStudyPreparation {
    pub compiled: CompiledAnalysisStudy,
    pub structural: StructuralPrestressedFrequencyStudyPreparation,
}

pub enum ThermalPrestressStructuralCompletion {
    Buckling(Box<CompletedStructuralBucklingSequenceStudy>),
    Modal(Box<CompletedStructuralPrestressedModalStudy>),
    Frequency(Box<CompletedStructuralPrestressedFrequencyStudy>),
    ModalFrequency(Box<CompletedStructuralPrestressedModalFrequencyStudy>),
}

pub struct CompletedThermalPrestressStudy {
    pub thermal: CompletedThermalStudy,
    pub structural: ThermalPrestressStructuralCompletion,
}

pub async fn execute_thermal_prestress_study_locally_with_source(
    mut config: NativeExecutionConfig,
    preparation: ThermalPrestressStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<(
    CompletedThermalPrestressStudy,
    crate::NativeProgramObjectSource,
)> {
    policy.validate()?;
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    let mut pool = LocalAnalysisTaskPool::new(session);
    let completed = execute_with_pool(&mut pool, preparation, policy).await?;
    Ok((completed, pool.object_source()))
}

pub async fn execute_thermal_prestress_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: ThermalPrestressStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalPrestressStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: ThermalPrestressStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedThermalPrestressStudy> {
    policy.validate()?;
    let ThermalPrestressStudyPreparation {
        compiled,
        mut structural,
    } = preparation;
    compiled
        .validate_against(&structural.mesh)
        .map_err(protocol)?;
    structural.workloads.validate().map_err(protocol)?;
    if structural.workloads.authored_model().map_err(protocol)? != &compiled.model
        || structural.authored_study_digest != compiled.definition_digest().map_err(protocol)?
        || structural.accelerator_requests != policy.resources.accelerators
    {
        return Err(protocol(
            "thermal prestress preparation differs from the exact compiled execution context",
        ));
    }
    let prefix = structural.workloads.prestress_prefix().map_err(protocol)?;
    let thermal = prefix
        .thermal
        .ok_or_else(|| protocol("thermal prestress composition requires a thermal producer"))?
        .clone();
    let [input] = prefix
        .preload
        .procedure_step()
        .map_err(protocol)?
        .state_inputs
        .as_slice()
    else {
        return Err(protocol("thermal preload requires exactly one named state"));
    };
    let StateSource::Upstream { state_id, .. } = &input.source else {
        return Err(protocol("thermal preload requires an upstream publication"));
    };
    let state_id = state_id.clone();
    let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = &structural.workloads
    else {
        return Err(protocol(
            "thermal prestress requires a heterogeneous sequence",
        ));
    };
    let consumers: Vec<_> = prefix
        .consumer_step_ids
        .iter()
        .map(|id| &workloads[id])
        .collect();
    let branch = match consumers.as_slice() {
        [StructuralStepWorkload::Buckling { .. }] => dispatch::Branch::Buckling,
        [StructuralStepWorkload::PrestressedModal { .. }] => dispatch::Branch::Modal,
        [StructuralStepWorkload::PrestressedFrequency { .. }] => dispatch::Branch::Frequency,
        [StructuralStepWorkload::PrestressedModal { .. }, StructuralStepWorkload::PrestressedFrequency { .. }] => {
            dispatch::Branch::ModalFrequency
        }
        _ => return Err(protocol("unsupported thermal prestress consumer inventory")),
    };
    let output_policy =
        AnalysisOutputPolicy::new(structural.chunk_bytes, structural.inventory_limits)
            .map_err(protocol)?;
    structural.chunk_bytes = output_policy.chunk_bytes();
    // Each phase owns a substantial state machine. Keep those child futures on
    // the heap instead of embedding their maximum frame in this composition.
    let executed = Box::pin(crate::analysis_thermal::execute_with_pool(
        pool,
        ThermalStudyPreparation {
            maximum_resident_bytes: Some(policy.resources.memory_bytes),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            field_source: None,
            scope: ThermalStudyScope::SelectedStep,
            product: ThermalStudyProduct {
                compiled,
                workload: thermal,
            },
            mesh: structural.mesh.clone(),
            frozen_project_digest: structural.frozen_project_digest,
            artifact_access: structural.artifact_access.clone(),
            program_revision: structural.program_revision.clone(),
            output_policy,
            required_capabilities: structural.required_capabilities.clone(),
        },
        ThermalStudyExecutionPolicy {
            resources: policy.resources.clone(),
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
        },
    ))
    .await?;
    let source = ThermalPrestressSource::import(
        pool,
        executed.completed.preparation(),
        executed.completed.result_closures().to_vec(),
    )
    .map_err(protocol)?;
    let dependency = *executed
        .state_publications
        .get(&state_id)
        .ok_or_else(|| protocol("selected thermal state has no completed scheduler task"))?;
    if dependency.identity
        != source
            .state_publication_dependency(&state_id)
            .map_err(protocol)?
    {
        return Err(protocol(
            "actual thermal publication task differs from sealed source lineage",
        ));
    }
    let bound = BoundThermalExecution {
        source,
        dependency,
        scope_id: executed.scope_id,
        pool_id: executed.pool_id,
    };
    let structural = Box::pin(dispatch::execute(pool, branch, structural, policy, bound)).await?;
    Ok(CompletedThermalPrestressStudy {
        thermal: executed.completed,
        structural,
    })
}

/// Constructed only after real thermal execution and authenticated replay on this pool.
pub(crate) struct BoundThermalExecution {
    source: ThermalPrestressSource,
    dependency: AnalysisTaskDependency,
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
}

impl BoundThermalExecution {
    pub(crate) fn into_parts(
        self,
        pool: &impl AnalysisTaskPool,
    ) -> NativeExecutionResult<(ThermalPrestressSource, AnalysisTaskDependency)> {
        if self.scope_id != pool.scope_id() || self.pool_id != pool.pool_id() {
            return Err(protocol(
                "thermal publication dependency belongs to another execution pool or scope",
            ));
        }
        Ok((self.source, self.dependency))
    }
}

pub(crate) fn validate_preload_dependencies(
    expected: &[AnalysisTaskIdentity],
    actual: &[AnalysisTaskDependency],
) -> NativeExecutionResult<()> {
    if expected
        != actual
            .iter()
            .map(|dependency| dependency.identity)
            .collect::<Vec<_>>()
    {
        return Err(protocol(
            "preload scheduler dependencies differ from the admitted thermal source",
        ));
    }
    Ok(())
}
