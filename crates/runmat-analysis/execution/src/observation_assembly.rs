use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ObservationAssemblyRequest,
    ObservationEvaluationRequest,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_assembly_resource_plan, observation_artifact_payload_bytes_upper_bound,
    FeaResourcePlan,
};
use runmat_analysis_fea::results::observation::{
    ObservationArtifact, ObservationArtifactInput, ObservationArtifactManifest, ObservationResult,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    analysis_object_references, import_analysis_output_set, import_analysis_partition,
    prepare_analysis_objects, prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisPartitionImport, ImportedObservations, ImportedReducedObservation,
    PreparedAnalysisObjects, PreparedAnalysisProgram,
};

use super::observation_partition::prepare_host_workload;
use crate::AnalysisHostWorkload;

pub struct ObservationAssemblyPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_request: ObservationEvaluationRequest,
    pub results: Vec<ImportedReducedObservation>,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedObservationAssembly {
    pub source_request: ObservationEvaluationRequest,
    pub request: ObservationAssemblyRequest,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

pub fn prepare_observation_assembly(
    preparation: ObservationAssemblyPreparation,
) -> AnalysisBridgeResult<PreparedObservationAssembly> {
    preparation.source_host.validate()?;
    preparation.source_request.validate()?;
    if preparation.chunk_bytes == 0 || preparation.results.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "observation assembly requires reduced results and bounded chunks".into(),
        ));
    }
    let mut by_observation = BTreeMap::new();
    for imported in preparation.results {
        if by_observation
            .insert(imported.result.observation_id.clone(), imported)
            .is_some()
        {
            return Err(AnalysisBridgeError::Identity(
                "observation assembly contains a duplicate result",
            ));
        }
    }
    let results = preparation
        .source_request
        .plan
        .observations
        .iter()
        .filter_map(|observation| by_observation.remove(&observation.spec.observation_id))
        .collect::<Vec<_>>();
    if !by_observation.is_empty() {
        return Err(AnalysisBridgeError::Identity(
            "observation assembly contains an unrelated or duplicate result",
        ));
    }
    let result_values = results
        .iter()
        .map(|imported| imported.result.clone())
        .collect::<Vec<_>>();
    let request = ObservationAssemblyRequest::new(&preparation.source_request, &result_values)?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationAssemblyRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let source_request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
        preparation.source_request.identity().digest(),
        &preparation.source_request.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = vec![request_objects, source_request_objects];
    inputs.extend(results.into_iter().map(|imported| imported.objects));
    let input_facts = crate::prepared_input_resource_facts(&inputs)?;
    let output_payload_upper_bound =
        observation_artifact_payload_bytes_upper_bound(&input_facts, result_values.len())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = build_observation_assembly_resource_plan(
        &preparation.source_request,
        &request,
        &result_values,
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
    inputs.push(resource_plan_objects.clone());
    let host = prepare_host_workload(
        &preparation.source_host,
        AnalysisWorkKind::AssembleObservations,
        &inputs,
        vec![AnalysisNumericalObjectKind::Observation],
        BTreeSet::new(),
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedObservationAssembly {
        source_request: preparation.source_request,
        request,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs,
    })
}

pub fn import_observation_assembly(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedObservationAssembly,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedObservations> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "observation assembly must return one observation root".into(),
        ));
    };
    let imported = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root: &output.root,
            expected_kind: AnalysisNumericalObjectKind::Observation,
            expected_logical_identity: output.logical_identity,
            expected_payload_digest: output.payload_digest,
            partition_identity: output.logical_identity,
            access: &prepared.host.artifact_access,
            limits,
        },
    )?;
    let mut expected = analysis_object_references(
        &imported.manifest,
        &output.root,
        &prepared.host.artifact_access,
        ValueRefKind::ResultObject,
    )?;
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "assembled observation inventory differs from its canonical closure",
        ));
    }
    let manifest = ObservationArtifactManifest::from_canonical_bytes(&imported.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let results = prepared
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ObservationResult)
        .map(|input| {
            ObservationResult::from_canonical_bytes(&input.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let expected_artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: prepared.source_request.identity(),
        producer: prepared.source_request.producer,
        source: prepared.source_request.selection.source(),
        source_selection: prepared.source_request.selection.identity(),
        topology_digest: prepared.source_request.plan.topology_digest,
        plan_identity: prepared.source_request.plan.identity,
        results,
    })
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_manifest = expected_artifact
        .manifest()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if manifest.identity.digest() != output.logical_identity
        || manifest.evaluation_request != prepared.source_request.identity()
        || manifest.producer != prepared.source_request.producer
        || manifest.source_selection != prepared.source_request.selection.identity()
        || manifest.source != prepared.source_request.selection.source()
        || manifest.topology_digest != prepared.source_request.plan.topology_digest
        || manifest.plan_identity != prepared.source_request.plan.identity
        || manifest != expected_manifest
    {
        return Err(AnalysisBridgeError::Identity(
            "assembled observations differ from their admitted evaluation request",
        ));
    }
    Ok(ImportedObservations {
        manifest,
        catalog_objects: imported,
        output: output.clone(),
        access: prepared.host.artifact_access.clone(),
    })
}
