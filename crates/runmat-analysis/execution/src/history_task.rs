use std::collections::BTreeSet;

use runmat_analysis_core::ObservationId;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, HistoryCheckpointRequest,
};
use runmat_analysis_fea::results::observation::{
    ObservationHistoryCheckpoint, ObservationResult, ResolvedObservationPlan,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    decode_prepared_observation_history_checkpoint, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_objects, prepare_analysis_program,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct HistoryCheckpointPreparation {
    pub source_host: AnalysisHostWorkload,
    pub plan: ResolvedObservationPlan,
    pub observation_id: ObservationId,
    pub sample: PreparedAnalysisObjects,
    pub previous: Option<PreparedAnalysisObjects>,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedHistoryCheckpointTask {
    pub request: HistoryCheckpointRequest,
    pub expected: ObservationHistoryCheckpoint,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedHistoryCheckpointTask {
    pub checkpoint: ObservationHistoryCheckpoint,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_history_checkpoint_task(
    preparation: HistoryCheckpointPreparation,
) -> AnalysisBridgeResult<PreparedHistoryCheckpointTask> {
    preparation.source_host.validate()?;
    if preparation.chunk_bytes == 0
        || preparation.sample.manifest.kind != AnalysisNumericalObjectKind::ObservationResult
        || preparation.previous.as_ref().is_some_and(|previous| {
            previous.manifest.kind != AnalysisNumericalObjectKind::Checkpoint
        })
    {
        return Err(AnalysisBridgeError::Invalid(
            "history checkpoint task requires one observation result and an optional checkpoint"
                .into(),
        ));
    }
    let sample = ObservationResult::from_canonical_bytes(&preparation.sample.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let previous = preparation
        .previous
        .as_ref()
        .map(decode_prepared_observation_history_checkpoint)
        .transpose()?;
    let predecessor = previous
        .as_ref()
        .map(|checkpoint| checkpoint.manifest().identity);
    let request = HistoryCheckpointRequest::new(
        preparation.plan,
        preparation.observation_id,
        predecessor,
        sample.logical_identity,
    )?;
    let expected = request.advance(previous, sample)?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::HistoryCheckpointRequest,
        request.identity.digest(),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut typed_inputs = vec![(request_objects, ValueRefKind::DriverObject)];
    typed_inputs.push((preparation.sample, ValueRefKind::ResultObject));
    if let Some(previous) = preparation.previous {
        typed_inputs.push((previous, ValueRefKind::ResultObject));
    }
    let workload_inputs = typed_inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let inputs = typed_inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let workload = AnalysisWorkloadRequest::new(
        preparation.source_host.workload.study,
        preparation.source_host.workload.step,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::CheckpointHistory,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::Checkpoint],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.source_host.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedHistoryCheckpointTask {
        request,
        expected,
        host,
        program,
        inputs,
    })
}

pub fn import_history_checkpoint_task(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedHistoryCheckpointTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedHistoryCheckpointTask> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "history checkpoint task must return exactly one checkpoint".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::Checkpoint,
        prepared.expected.manifest().identity.digest(),
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let checkpoint = decode_prepared_observation_history_checkpoint(&objects)?;
    if checkpoint != prepared.expected {
        return Err(AnalysisBridgeError::Identity(
            "published history checkpoint differs from its prepared transition",
        ));
    }
    let mut expected_objects =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected_objects.push(output_set.root);
    if output_set.result_objects != expected_objects {
        return Err(AnalysisBridgeError::Identity(
            "history checkpoint result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedHistoryCheckpointTask {
        checkpoint,
        objects,
    })
}
