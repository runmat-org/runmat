use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ResultPublicationRequest,
};
use runmat_analysis_fea::planning::resource::{
    validate_resource_usage_evidence, FeaResourcePlan, ResourceUsageEvidence,
};
use runmat_analysis_fea::results::accuracy::AccuracyArtifact;
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, validate_observation_resource_usage, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, ImportedObservations, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct ResultPublicationPreparation {
    pub accuracy_host: AnalysisHostWorkload,
    pub source_accuracy: PreparedAnalysisObjects,
    pub source_resource_plan: PreparedAnalysisObjects,
    pub source_resource_usage: PreparedAnalysisObjects,
    pub source_observations: Option<ImportedObservations>,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedResultPublication {
    pub request: ResultPublicationRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedPublishedResult {
    pub result: PublishedAnalysisResult,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_result_publication(
    preparation: ResultPublicationPreparation,
) -> AnalysisBridgeResult<PreparedResultPublication> {
    preparation.accuracy_host.validate()?;
    if preparation.accuracy_host.workload.kind != AnalysisWorkKind::EvaluateAccuracy
        || preparation.accuracy_host.workload.outputs
            != vec![AnalysisNumericalObjectKind::AccuracyReport]
        || preparation.source_accuracy.manifest.kind != AnalysisNumericalObjectKind::AccuracyReport
        || preparation.source_resource_plan.manifest.kind
            != AnalysisNumericalObjectKind::ResourcePlan
        || preparation.source_resource_usage.manifest.kind
            != AnalysisNumericalObjectKind::ResourceUsage
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "result publication requires one completed accuracy evaluation and bounded chunks"
                .into(),
        ));
    }
    let accuracy = AccuracyArtifact::from_canonical_bytes(&preparation.source_accuracy.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan =
        FeaResourcePlan::from_canonical_bytes(&preparation.source_resource_plan.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_usage =
        ResourceUsageEvidence::from_canonical_bytes(&preparation.source_resource_usage.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    validate_resource_usage_evidence(&resource_plan, &resource_usage)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan.identity.digest() != preparation.source_resource_plan.manifest.logical_identity
        || resource_usage.identity().digest()
            != preparation.source_resource_usage.manifest.logical_identity
        || resource_usage.result_digest != accuracy.source_result.digest()
        || !resource_usage.overruns.is_empty()
    {
        return Err(AnalysisBridgeError::Invalid(
            "result publication requires reconciled source-bound resource usage without overruns"
                .into(),
        ));
    }
    if accuracy.identity().digest() != preparation.source_accuracy.manifest.logical_identity
        || accuracy.producer != preparation.accuracy_host.workload.step
        || !preparation
            .accuracy_host
            .workload
            .inputs
            .iter()
            .any(|input| {
                input.kind == AnalysisNumericalObjectKind::AccuracyEvaluationRequest
                    && input.logical_identity == accuracy.evaluation_request.digest()
            })
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy artifact differs from its evaluation workload",
        ));
    }
    let observations = preparation
        .source_observations
        .as_ref()
        .map(|observations| {
            if observations.output.kind != AnalysisNumericalObjectKind::Observation {
                return Err(AnalysisBridgeError::Invalid(
                    "publication observation input has the wrong numerical kind".into(),
                ));
            }
            observations
                .manifest
                .validate()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            validate_observation_resource_usage(&observations.manifest, &resource_usage)?;
            if observations.manifest.identity.digest() != observations.output.logical_identity {
                return Err(AnalysisBridgeError::Identity(
                    "publication observation manifest differs from its content",
                ));
            }
            Ok(observations.manifest.clone())
        })
        .transpose()?;
    let request = ResultPublicationRequest::new(
        &accuracy,
        observations.as_ref(),
        &resource_plan,
        &resource_usage,
    )?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::PublicationRequest,
        request.identity().digest(),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (preparation.source_accuracy, ValueRefKind::ResultObject),
        (preparation.source_resource_plan, ValueRefKind::DriverObject),
        (
            preparation.source_resource_usage,
            ValueRefKind::ResultObject,
        ),
    ];
    if let Some(observations) = preparation.source_observations {
        let bytes = observations
            .manifest
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        inputs.push((
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::ObservationCatalog,
                observations.manifest.identity.digest(),
                &bytes,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )?,
            ValueRefKind::DriverObject,
        ));
    }
    let workload_inputs = inputs
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
    let inputs = inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let workload = AnalysisWorkloadRequest::new(
        preparation.accuracy_host.workload.study,
        preparation.accuracy_host.workload.step,
        preparation.accuracy_host.workload.execution_plan,
        AnalysisWorkKind::PublishResult,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::PublishedResult],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.accuracy_host.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedResultPublication {
        request,
        host,
        program,
        inputs,
    })
}

pub fn import_published_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedResultPublication,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedPublishedResult> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "result publication must return exactly one published-result root".into(),
        ));
    };
    let imported = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::PublishedResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        imported.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "published result inventory differs from its canonical closure",
        ));
    }
    let result = PublishedAnalysisResult::from_canonical_bytes(&imported.payload())?;
    if result.identity().digest() != output.logical_identity
        || result.request != prepared.request.identity()
        || result.producer != prepared.request.producer
        || result.source_result != prepared.request.source_result
        || result.source_observations != prepared.request.source_observations
        || result.source_accuracy != prepared.request.source_accuracy
        || result.source_resource_plan != prepared.request.source_resource_plan
        || result.source_resource_usage != prepared.request.source_resource_usage
    {
        return Err(AnalysisBridgeError::Identity(
            "published result differs from its admitted request",
        ));
    }
    Ok(ImportedPublishedResult {
        result,
        objects: imported,
    })
}
