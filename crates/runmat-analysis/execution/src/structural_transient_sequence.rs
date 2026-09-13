use std::collections::BTreeMap;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisTaskIdentity, StructuralTransientBlockRequest,
    StructuralTransientBlockRequestIdentity, StructuralTransientCheckpointScope,
    StructuralTransientResultAssemblyRequestIdentity,
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
use runmat_meshing_core::CanonicalMeshingContract;

use crate::{
    decode_prepared_structural_transient_workload, import_structural_transient_block,
    import_structural_transient_result, prepare_structural_transient_block_request_objects,
    prepare_structural_transient_block_task, prepare_structural_transient_checkpoint_objects,
    prepare_structural_transient_result_task, AnalysisBridgeError, AnalysisBridgeResult,
    ImportedStructuralTransientBlock, ImportedStructuralTransientResult, PreparedAnalysisObjects,
    PreparedStructuralTransientBlockTask, PreparedStructuralTransientResultTask,
    StructuralTransientBlockPreparation, StructuralTransientExecutionCheckpoint,
    StructuralTransientRecoveryClosure, StructuralTransientResultPreparation,
    STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA,
};

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransientSequencePreparation {
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub scope: StructuralTransientCheckpointScope,
    pub terminal_schedule_ordinals: Vec<u64>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
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
                workload: self.preparation.workload.clone(),
                mesh: self.preparation.mesh.clone(),
                request: request_objects,
                predecessor: predecessor_objects,
                artifact_access: self.preparation.artifact_access.clone(),
                program_revision: self.preparation.program_revision.clone(),
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
        let dependency_identities = self
            .completed_block_tasks
            .last()
            .copied()
            .into_iter()
            .collect::<Vec<_>>();
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
        self.completed_block_tasks
            .last()
            .copied()
            .into_iter()
            .collect()
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
    preparation.scope.identity()?;
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
    {
        return Err(invalid(
            "transient sequence preparation has invalid scope, workload, or block boundaries",
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
