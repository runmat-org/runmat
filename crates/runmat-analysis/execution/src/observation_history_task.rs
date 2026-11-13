use std::collections::BTreeSet;

use runmat_analysis_core::ObservationId;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ObservationHistoryCheckpointRequest,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_history_checkpoint_resource_plan,
    observation_history_checkpoint_payload_bytes_upper_bound, FeaResourcePlan,
};
use runmat_analysis_fea::results::observation::{
    ObservationHistoryCheckpoint, ResolvedObservationPlan,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    decode_prepared_observation_history_checkpoint, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_objects, prepare_analysis_program,
    prepared_input_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, ImportedReducedObservation, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct ObservationHistoryCheckpointPreparation {
    pub source_host: AnalysisHostWorkload,
    pub plan: ResolvedObservationPlan,
    pub observation_id: ObservationId,
    pub sample: ImportedReducedObservation,
    pub previous: Option<PreparedAnalysisObjects>,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedObservationHistoryCheckpointTask {
    pub request: ObservationHistoryCheckpointRequest,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservationHistoryCheckpointTask {
    pub checkpoint: ObservationHistoryCheckpoint,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_observation_history_checkpoint_task(
    preparation: ObservationHistoryCheckpointPreparation,
) -> AnalysisBridgeResult<PreparedObservationHistoryCheckpointTask> {
    preparation.source_host.validate()?;
    if preparation.chunk_bytes == 0
        || preparation.source_host.workload.kind != AnalysisWorkKind::ReduceObservationPartitions
        || preparation.source_host.workload.outputs
            != vec![AnalysisNumericalObjectKind::ObservationResult]
        || preparation.sample.objects.manifest.kind
            != AnalysisNumericalObjectKind::ObservationResult
        || preparation.sample.producer != preparation.source_host.workload.identity
        || preparation.previous.as_ref().is_some_and(|previous| {
            previous.manifest.kind != AnalysisNumericalObjectKind::ObservationHistoryCheckpoint
        })
    {
        return Err(AnalysisBridgeError::Invalid(
            "history checkpoint task requires one reduced observation result, its reduction host, and an optional checkpoint".into(),
        ));
    }
    let sample = preparation.sample.result.clone();
    let previous = preparation
        .previous
        .as_ref()
        .map(decode_prepared_observation_history_checkpoint)
        .transpose()?;
    let predecessor = previous
        .as_ref()
        .map(|checkpoint| checkpoint.manifest().identity);
    let sample_ordinal = previous
        .as_ref()
        .map_or(0, |checkpoint| checkpoint.manifest().observed_sample_count);
    let request = ObservationHistoryCheckpointRequest::new(
        preparation.plan,
        preparation.source_host.workload.step,
        preparation.observation_id,
        predecessor,
        sample_ordinal,
        &sample,
    )?;
    request.validate_advance_inputs(previous.as_ref(), &sample)?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationHistoryCheckpointRequest,
        request.identity.digest(),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut typed_inputs = vec![(request_objects, ValueRefKind::DriverObject)];
    typed_inputs.push((preparation.sample.objects, ValueRefKind::ResultObject));
    if let Some(previous) = preparation.previous {
        typed_inputs.push((previous, ValueRefKind::ResultObject));
    }
    let numerical_inputs = typed_inputs
        .iter()
        .map(|(input, _)| input.clone())
        .collect::<Vec<_>>();
    let input_facts = prepared_input_resource_facts(&numerical_inputs)?;
    let output_payload_upper_bound = observation_history_checkpoint_payload_bytes_upper_bound(
        &request,
        previous.as_ref(),
        &sample,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = build_observation_history_checkpoint_resource_plan(
        &request,
        previous.as_ref(),
        &sample,
        input_facts,
        crate::projected_output_resource_facts(output_payload_upper_bound)?,
        crate::projected_output_set_resource_facts(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    typed_inputs.push((resource_plan_objects.clone(), ValueRefKind::DriverObject));
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
        request.producer,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::CheckpointObservationHistory,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::ObservationHistoryCheckpoint],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        workload,
        preparation.source_host.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedObservationHistoryCheckpointTask {
        request,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs,
    })
}

pub fn import_observation_history_checkpoint_task(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedObservationHistoryCheckpointTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedObservationHistoryCheckpointTask> {
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
        AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let checkpoint = decode_prepared_observation_history_checkpoint(&objects)?;
    let sample = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ObservationResult)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("history checkpoint lost its sample input".into())
        })?;
    let sample =
        runmat_analysis_fea::results::observation::ObservationResult::from_canonical_bytes(
            &sample.payload(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let previous = prepared
        .inputs
        .iter()
        .find(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::ObservationHistoryCheckpoint
        })
        .map(decode_prepared_observation_history_checkpoint)
        .transpose()?;
    let expected = prepared.request.advance(previous, sample)?;
    if checkpoint != expected || output.logical_identity != expected.manifest().identity.digest() {
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
    Ok(ImportedObservationHistoryCheckpointTask {
        checkpoint,
        objects,
    })
}
