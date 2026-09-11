use std::collections::BTreeMap;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ObservationAssemblyRequest,
    ObservationEvaluationRequest,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_analysis_fea::results::observation::ObservationArtifactManifest;
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

pub struct StructuralObservationAssemblyPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_request: ObservationEvaluationRequest,
    pub results: Vec<ImportedReducedObservation>,
    pub expected_resource_plan: FeaResourcePlan,
    pub resource_plan: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralObservationAssembly {
    pub source_request: ObservationEvaluationRequest,
    pub request: ObservationAssemblyRequest,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

pub fn prepare_structural_observation_assembly(
    preparation: StructuralObservationAssemblyPreparation,
) -> AnalysisBridgeResult<PreparedStructuralObservationAssembly> {
    preparation.source_host.validate()?;
    preparation.source_request.validate()?;
    if preparation.chunk_bytes == 0 || preparation.results.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "observation assembly requires reduced results and bounded chunks".into(),
        ));
    }
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&preparation.resource_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan != preparation.expected_resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "observation assembly received a substituted resource plan",
        ));
    }
    let mut by_observation = preparation
        .results
        .into_iter()
        .map(|imported| (imported.result.observation_id.clone(), imported))
        .collect::<BTreeMap<_, _>>();
    let results = preparation
        .source_request
        .plan
        .observations
        .iter()
        .map(|observation| {
            by_observation
                .remove(&observation.spec.observation_id)
                .ok_or(AnalysisBridgeError::Identity(
                    "observation assembly is missing a resolved result",
                ))
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    if !by_observation.is_empty() {
        return Err(AnalysisBridgeError::Identity(
            "observation assembly contains an unrelated or duplicate result",
        ));
    }
    let request = ObservationAssemblyRequest::new(
        &preparation.source_request,
        &results
            .iter()
            .map(|imported| imported.result.clone())
            .collect::<Vec<_>>(),
        &resource_plan,
    )?;
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
    inputs.push(preparation.resource_plan.clone());
    let host = prepare_host_workload(
        &preparation.source_host,
        AnalysisWorkKind::AssembleObservations,
        &inputs,
        vec![AnalysisNumericalObjectKind::Observation],
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralObservationAssembly {
        source_request: preparation.source_request,
        request,
        resource_plan,
        resource_plan_objects: preparation.resource_plan,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_observation_assembly(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralObservationAssembly,
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
    if manifest.identity.digest() != output.logical_identity
        || manifest.evaluation_request != prepared.source_request.identity()
        || manifest.producer != prepared.source_request.producer
        || manifest.field_recovery_request != prepared.source_request.field_recovery_request
        || manifest.source_fields != prepared.source_request.source_fields
        || manifest.source_result != prepared.source_request.source_result
        || manifest.topology_digest != prepared.source_request.plan.topology_digest
        || manifest.plan_identity != prepared.source_request.plan.identity
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
