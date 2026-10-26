use std::collections::BTreeMap;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisTaskIdentity, CanonicalSemanticState,
    StructuralTransientBlockRequest, StructuralTransientBlockRequestIdentity,
    StructuralTransientCheckpointScope, StructuralTransientCoordinateSpace,
    StructuralTransientResultAssemblyRequestIdentity, StructuralTransientSolverPlan,
    StructuralTransientWorkload,
};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    decode_prepared_structural_transient_workload, import_structural_transient_block,
    import_structural_transient_result, object::rebind_unpartitioned_analysis_identity,
    prepare_structural_transient_block_request_objects, prepare_structural_transient_block_task,
    prepare_structural_transient_checkpoint_objects, prepare_structural_transient_result_task,
    AnalysisBridgeError, AnalysisBridgeResult, ImportedStructuralTransientBlock,
    ImportedStructuralTransientResult, ImportedStructuralTransientScope, PreparedAnalysisObjects,
    PreparedStructuralTransientBlockTask, PreparedStructuralTransientResultTask,
    PreparedStructuralTransientScopeTask, StructuralTransientBlockPlan,
    StructuralTransientBlockPreparation, StructuralTransientExecutionCheckpoint,
    StructuralTransientRecoveryClosure, StructuralTransientResultPreparation,
    STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA,
};

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientSequencePreparation {
    /// Admitted local execution, read from the scope task's declared resource plan.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub modal_state: Option<PreparedAnalysisObjects>,
    pub scope: StructuralTransientCheckpointScope,
    pub terminal_schedule_ordinals: Vec<u64>,
    /// Logical task predecessors that produced immutable inputs consumed by the first block.
    /// Later blocks inherit them transitively through the preceding block task.
    pub initial_dependencies: Vec<AnalysisTaskIdentity>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct StructuralTransientSequencePreparationInput<'a> {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub workload: &'a StructuralTransientWorkload,
    pub mesh: &'a SolverMeshArtifact,
    pub modal_state: Option<&'a CanonicalSemanticState>,
    pub block_plan: StructuralTransientBlockPlan,
    pub initial_dependencies: Vec<AnalysisTaskIdentity>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientSequencePlan {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub workload: StructuralTransientWorkload,
    pub mesh: SolverMeshArtifact,
    pub modal_state: Option<CanonicalSemanticState>,
    pub block_plan: StructuralTransientBlockPlan,
    pub initial_dependencies: Vec<AnalysisTaskIdentity>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub fn prepare_structural_transient_sequence(
    input: StructuralTransientSequencePreparationInput<'_>,
) -> AnalysisBridgeResult<StructuralTransientSequencePlan> {
    input.workload.validate()?;
    input
        .mesh
        .validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    input.artifact_access.validate()?;
    if input.chunk_bytes == 0 {
        return Err(invalid("transient sequence requires a nonzero chunk size"));
    }
    let mut unique_dependencies = input.initial_dependencies.clone();
    unique_dependencies.sort_unstable();
    let original_count = unique_dependencies.len();
    unique_dependencies.dedup();
    if unique_dependencies.len() != original_count {
        return Err(invalid(
            "transient sequence has duplicate initial dependencies",
        ));
    }
    Ok(StructuralTransientSequencePlan {
        local_execution: input.local_execution,
        workload: input.workload.clone(),
        mesh: input.mesh.clone(),
        modal_state: input.modal_state.cloned(),
        block_plan: input.block_plan,
        initial_dependencies: input.initial_dependencies,
        artifact_access: input.artifact_access,
        program_revision: input.program_revision,
        chunk_bytes: input.chunk_bytes,
        inventory_limits: input.inventory_limits,
    })
}

pub fn complete_structural_transient_sequence_preparation(
    plan: StructuralTransientSequencePlan,
    prepared_scope: PreparedStructuralTransientScopeTask,
    imported_scope: ImportedStructuralTransientScope,
    scope_task_identity: AnalysisTaskIdentity,
) -> AnalysisBridgeResult<StructuralTransientSequencePreparation> {
    let expected_task_identity = AnalysisTaskIdentity::new(
        prepared_scope.host.workload.identity,
        plan.initial_dependencies.clone(),
    )?;
    let scope = &imported_scope.scope;
    let workload_digest = Digest::sha256(plan.workload.canonical_bytes()?);
    let modal_identity = plan
        .modal_state
        .as_ref()
        .map(|state| state.identity.digest());
    let prepared_modal_identity = prepared_scope
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SemanticState)
        .map(|input| input.manifest.logical_identity);
    if scope_task_identity != expected_task_identity
        || prepared_scope.host.artifact_access != plan.artifact_access
        || prepared_scope
            .inputs
            .first()
            .is_none_or(|input| input.manifest.payload_digest != workload_digest)
        || prepared_scope.inputs.get(1).is_none_or(|input| {
            input.manifest.logical_identity
                != Digest::from_bytes(*plan.mesh.canonical_digest.bytes())
        })
        || prepared_modal_identity != modal_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "transient scope task differs from the planned sequence inputs or dependencies",
        ));
    }
    let terminal_schedule_ordinal = u64::try_from(scope.schedule.points.len() - 1)
        .map_err(|_| invalid("transient schedule exceeds u64 ordinals"))?;
    let terminal_schedule_ordinals = plan
        .block_plan
        .terminal_schedule_ordinals(terminal_schedule_ordinal)?;
    let mut scope_inputs = prepared_scope.inputs.into_iter();
    let workload = scope_inputs.next().ok_or(AnalysisBridgeError::Identity(
        "transient scope task omitted its workload input",
    ))?;
    let workload = rebind_unpartitioned_analysis_identity(
        workload,
        scope.source_workload.digest(),
        plan.inventory_limits,
    )?;
    let mesh = scope_inputs.next().ok_or(AnalysisBridgeError::Identity(
        "transient scope task omitted its mesh input",
    ))?;
    let modal_state = if plan.modal_state.is_some() {
        Some(scope_inputs.next().ok_or(AnalysisBridgeError::Identity(
            "modal transient scope task omitted its basis input",
        ))?)
    } else {
        None
    };
    let resource_input = scope_inputs.next().ok_or(AnalysisBridgeError::Identity(
        "transient scope task omitted its resource-plan input",
    ))?;
    if resource_input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan
        || resource_input.manifest.logical_identity
            != prepared_scope.resource_plan.identity.digest()
        || resource_input.payload()
            != prepared_scope
                .resource_plan
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
        || scope_inputs.next().is_some()
    {
        return Err(AnalysisBridgeError::Identity(
            "transient scope task inputs differ from its admitted resource plan",
        ));
    }
    let mut initial_dependencies = plan.initial_dependencies;
    initial_dependencies.push(scope_task_identity);
    let local_execution =
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
            prepared_scope.resource_plan.request.cpu_millicores,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if local_execution != plan.local_execution {
        return Err(AnalysisBridgeError::Identity(
            "transient scope task admitted another local execution plan",
        ));
    }
    let preparation = StructuralTransientSequencePreparation {
        local_execution,
        workload,
        mesh,
        modal_state,
        scope: imported_scope.scope,
        terminal_schedule_ordinals,
        initial_dependencies,
        artifact_access: plan.artifact_access,
        program_revision: plan.program_revision,
        chunk_bytes: plan.chunk_bytes,
        inventory_limits: plan.inventory_limits,
    };
    validate_preparation(&preparation)?;
    Ok(preparation)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SequencePhase {
    ReadyForBlock,
    AwaitingBlock(StructuralTransientBlockRequestIdentity),
    ReadyForAssembly,
    AwaitingAssembly(StructuralTransientResultAssemblyRequestIdentity),
    Complete,
}

pub struct StructuralTransientSequenceCoordinator {
    preparation: StructuralTransientSequencePreparation,
    blocks: Vec<ImportedStructuralTransientBlock>,
    completed_block_responses: Vec<ProgramExecutionResponse>,
    completed_block_tasks: Vec<AnalysisTaskIdentity>,
    result: Option<ImportedStructuralTransientResult>,
    phase: SequencePhase,
}

impl StructuralTransientSequenceCoordinator {
    pub fn new(preparation: StructuralTransientSequencePreparation) -> AnalysisBridgeResult<Self> {
        validate_preparation(&preparation)?;
        Ok(Self {
            preparation,
            blocks: Vec::new(),
            completed_block_responses: Vec::new(),
            completed_block_tasks: Vec::new(),
            result: None,
            phase: SequencePhase::ReadyForBlock,
        })
    }

    pub fn prepare_next_block(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedStructuralTransientBlockTask> {
        if self.phase != SequencePhase::ReadyForBlock {
            return Err(invalid("transient sequence is not ready for another block"));
        }
        let terminal_schedule_ordinal = *self
            .preparation
            .terminal_schedule_ordinals
            .get(self.blocks.len())
            .ok_or_else(|| invalid("transient sequence has no remaining block boundary"))?;
        let predecessor = self
            .blocks
            .last()
            .map(|block| &block.segment.manifest().terminal_checkpoint);
        let request = StructuralTransientBlockRequest::new(
            self.preparation.scope.clone(),
            predecessor,
            terminal_schedule_ordinal,
        )?;
        let request_objects = prepare_structural_transient_block_request_objects(
            &request,
            self.preparation.chunk_bytes,
            ObjectNamespace::InputValue,
            self.preparation.inventory_limits,
        )?;
        let predecessor_objects = predecessor
            .map(|checkpoint| {
                prepare_structural_transient_checkpoint_objects(
                    checkpoint,
                    self.preparation.chunk_bytes,
                    ObjectNamespace::ResultValue,
                    self.preparation.inventory_limits,
                )
            })
            .transpose()?;
        let prepared =
            prepare_structural_transient_block_task(StructuralTransientBlockPreparation {
                local_execution: self.preparation.local_execution,
                workload: self.preparation.workload.clone(),
                mesh: self.preparation.mesh.clone(),
                request: request_objects,
                predecessor: predecessor_objects,
                modal_state: self.preparation.modal_state.clone(),
                artifact_access: self.preparation.artifact_access.clone(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: self.preparation.chunk_bytes,
                inventory_limits: self.preparation.inventory_limits,
            })?;
        self.phase = SequencePhase::AwaitingBlock(prepared.request.identity);
        Ok(prepared)
    }

    pub fn complete_block(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralTransientBlockTask,
    ) -> AnalysisBridgeResult<()> {
        let expected_block_ordinal = u64::try_from(self.blocks.len())
            .map_err(|_| invalid("transient sequence block count exceeds u64"))?;
        if self.phase != SequencePhase::AwaitingBlock(prepared.request.identity)
            || prepared.request.block_ordinal != expected_block_ordinal
            || prepared.request.scope != self.preparation.scope
            || self
                .preparation
                .terminal_schedule_ordinals
                .get(self.blocks.len())
                != Some(&prepared.request.terminal_schedule_ordinal)
        {
            return Err(AnalysisBridgeError::Identity(
                "transient block response differs from the awaited sequence transition",
            ));
        }
        let block = import_structural_transient_block(
            source,
            response.clone(),
            prepared,
            self.preparation.inventory_limits,
        )?;
        let dependency_identities = self.next_block_dependency_identities();
        let task_identity =
            AnalysisTaskIdentity::new(prepared.host.workload.identity, dependency_identities)?;
        self.blocks.push(block);
        self.completed_block_responses.push(response);
        self.completed_block_tasks.push(task_identity);
        self.phase = if self.blocks.len() == self.preparation.terminal_schedule_ordinals.len() {
            SequencePhase::ReadyForAssembly
        } else {
            SequencePhase::ReadyForBlock
        };
        Ok(())
    }

    pub fn prepare_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedStructuralTransientResultTask> {
        if self.phase != SequencePhase::ReadyForAssembly {
            return Err(invalid(
                "transient sequence is not at its terminal assembly boundary",
            ));
        }
        let prepared =
            prepare_structural_transient_result_task(StructuralTransientResultPreparation {
                blocks: self.blocks.clone(),
                artifact_access: self.preparation.artifact_access.clone(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: self.preparation.chunk_bytes,
                inventory_limits: self.preparation.inventory_limits,
            })?;
        self.phase = SequencePhase::AwaitingAssembly(prepared.request.identity);
        Ok(prepared)
    }

    pub fn complete_assembly(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralTransientResultTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != SequencePhase::AwaitingAssembly(prepared.request.identity) {
            return Err(AnalysisBridgeError::Identity(
                "transient assembly response differs from the awaited sequence transition",
            ));
        }
        self.result = Some(import_structural_transient_result(
            source,
            response,
            prepared,
            self.preparation.inventory_limits,
        )?);
        self.phase = SequencePhase::Complete;
        Ok(())
    }

    pub fn completed_blocks(&self) -> &[ImportedStructuralTransientBlock] {
        &self.blocks
    }

    pub fn completed_block_task_identities(&self) -> &[AnalysisTaskIdentity] {
        &self.completed_block_tasks
    }

    pub fn next_block_dependency_identities(&self) -> Vec<AnalysisTaskIdentity> {
        self.completed_block_tasks.last().copied().map_or_else(
            || self.preparation.initial_dependencies.clone(),
            |dependency| vec![dependency],
        )
    }

    pub fn assembly_dependency_identities(&self) -> Vec<AnalysisTaskIdentity> {
        self.completed_block_tasks.clone()
    }

    pub fn has_remaining_blocks(&self) -> bool {
        self.blocks.len() < self.preparation.terminal_schedule_ordinals.len()
    }

    pub fn checkpoint(&self) -> AnalysisBridgeResult<StructuralTransientExecutionCheckpoint> {
        if !matches!(
            self.phase,
            SequencePhase::ReadyForBlock | SequencePhase::ReadyForAssembly
        ) || self.completed_block_responses.is_empty()
        {
            return Err(invalid(
                "transient sequence checkpoint requires an accepted block boundary",
            ));
        }
        StructuralTransientExecutionCheckpoint::new(
            &self.preparation,
            self.completed_block_responses.clone(),
        )
    }

    pub fn recovery_closure(
        &self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<StructuralTransientRecoveryClosure> {
        self.checkpoint()?
            .prepare_closure(source, &self.preparation)
    }

    pub fn restore(
        preparation: StructuralTransientSequencePreparation,
        source: &impl CacheImport,
        checkpoint: &StructuralTransientExecutionCheckpoint,
    ) -> AnalysisBridgeResult<Self> {
        checkpoint.validate_for(&preparation)?;
        let responses = checkpoint.responses().to_vec();
        let mut coordinator = Self::new(preparation)?;
        for response in responses {
            let prepared = coordinator.prepare_next_block()?;
            coordinator.complete_block(source, response, &prepared)?;
        }
        let restored = coordinator.checkpoint()?;
        if restored != *checkpoint {
            return Err(AnalysisBridgeError::Identity(
                "restored transient sequence differs from its checkpoint",
            ));
        }
        Ok(coordinator)
    }

    pub fn restore_from_closure(
        preparation: StructuralTransientSequencePreparation,
        root: &ValueRef,
        objects: &[ProgramExecutionObject],
    ) -> AnalysisBridgeResult<Self> {
        preparation.artifact_access.validate_reference(
            root,
            ValueRefKind::ResultObject,
            crate::STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_MEDIA_TYPE,
            STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA,
        )?;
        let source = RecoveryObjectSource::new(objects)?;
        let checkpoint_bytes =
            source
                .read_verified(root.logical_digest)?
                .ok_or(AnalysisBridgeError::Identity(
                    "transient recovery closure omits its checkpoint root",
                ))?;
        let checkpoint =
            StructuralTransientExecutionCheckpoint::from_canonical_bytes(&checkpoint_bytes)?;
        let coordinator = Self::restore(preparation, &source, &checkpoint)?;
        let expected = coordinator.recovery_closure(&source)?;
        let mut supplied = objects.to_vec();
        supplied.sort_by_key(|object| object.reference.id);
        if expected.root != *root || expected.objects != supplied {
            return Err(AnalysisBridgeError::Identity(
                "transient recovery closure contains missing or unrelated objects",
            ));
        }
        Ok(coordinator)
    }

    pub fn is_complete(&self) -> bool {
        self.phase == SequencePhase::Complete
    }

    pub fn finish(self) -> AnalysisBridgeResult<ImportedStructuralTransientResult> {
        if self.phase != SequencePhase::Complete {
            return Err(invalid("transient sequence is incomplete"));
        }
        self.result
            .ok_or_else(|| invalid("completed transient sequence has no assembled result"))
    }
}

fn validate_preparation(
    preparation: &StructuralTransientSequencePreparation,
) -> AnalysisBridgeResult<()> {
    preparation.artifact_access.validate()?;
    let workload = decode_prepared_structural_transient_workload(&preparation.workload)?;
    if preparation.mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh {
        return Err(invalid(
            "transient sequence preparation requires one canonical mesh",
        ));
    }
    let mesh =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&preparation.mesh.payload())
            .map_err(|error| invalid(error.to_string()))?;
    let modal_state = preparation
        .modal_state
        .as_ref()
        .map(crate::decode_prepared_semantic_state)
        .transpose()?;
    preparation.scope.identity()?;
    let mut sorted_dependencies = preparation.initial_dependencies.clone();
    sorted_dependencies.sort_unstable();
    let unique_dependency_count = sorted_dependencies.len();
    sorted_dependencies.dedup();
    let terminal_schedule_ordinal = u64::try_from(
        preparation
            .scope
            .schedule
            .points
            .len()
            .checked_sub(1)
            .ok_or_else(|| invalid("transient sequence schedule is empty"))?,
    )
    .map_err(|_| invalid("transient sequence schedule exceeds the u64 ordinal range"))?;
    if preparation.chunk_bytes == 0
        || preparation.terminal_schedule_ordinals.is_empty()
        || preparation.terminal_schedule_ordinals.last() != Some(&terminal_schedule_ordinal)
        || preparation
            .terminal_schedule_ordinals
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || preparation.terminal_schedule_ordinals[0] == 0
        || preparation.scope.source_workload.digest()
            != preparation.workload.manifest.logical_identity
        || preparation.mesh.manifest.logical_identity
            != Digest::from_bytes(*mesh.canonical_digest.bytes())
        || workload.canonical_bytes()?.as_slice() != preparation.workload.payload().as_slice()
        || sorted_dependencies.len() != unique_dependency_count
    {
        return Err(invalid(
            "transient sequence preparation has invalid scope, workload, or block boundaries",
        ));
    }
    let modal_state_matches = match (
        &workload.solver,
        &preparation.scope.coordinate_space,
        modal_state.as_ref(),
    ) {
        (
            StructuralTransientSolverPlan::Direct,
            StructuralTransientCoordinateSpace::Direct,
            None,
        ) => true,
        (
            StructuralTransientSolverPlan::ModalReduced { basis_transition },
            StructuralTransientCoordinateSpace::ModalReduced {
                basis_state: scoped_basis,
                basis,
                coordinate_count,
                ..
            },
            Some(state),
        ) => {
            state.identity == basis_transition.source_state
                && state.identity == *scoped_basis
                && state.topology_digest == Digest::from_bytes(*mesh.canonical_digest.bytes())
                && state.payload.modal_basis().is_some_and(|state_basis| {
                    state_basis.identity() == *basis
                        && state_basis.modes().len() == *coordinate_count
                })
        }
        _ => false,
    };
    if !modal_state_matches {
        return Err(AnalysisBridgeError::Identity(
            "transient sequence modal state differs from its workload and checkpoint scope",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}

struct RecoveryObjectSource {
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl RecoveryObjectSource {
    fn new(objects: &[ProgramExecutionObject]) -> AnalysisBridgeResult<Self> {
        let mut by_id = BTreeMap::new();
        let mut by_digest = BTreeMap::new();
        for object in objects {
            object.validate()?;
            if object.reference.kind != ValueRefKind::ResultObject {
                return Err(invalid(
                    "transient recovery closure contains a non-result object",
                ));
            }
            if let Some(existing) = by_id.insert(object.reference.id, &object.reference) {
                if existing != &object.reference {
                    return Err(AnalysisBridgeError::Identity(
                        "transient recovery closure reuses an object id",
                    ));
                }
            }
            if let Some(existing) = by_digest.insert(object.reference.logical_digest, &object.bytes)
            {
                if existing != &object.bytes {
                    return Err(AnalysisBridgeError::Identity(
                        "transient recovery closure reuses a digest",
                    ));
                }
            }
        }
        Ok(Self {
            objects: by_digest
                .into_iter()
                .map(|(digest, bytes)| (digest, bytes.clone()))
                .collect(),
        })
    }
}

impl CacheImport for RecoveryObjectSource {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.objects.get(&digest).cloned())
    }
}
