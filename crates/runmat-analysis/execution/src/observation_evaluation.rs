use std::collections::BTreeSet;

use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, FieldRecoveryRequest,
    ObservationEvaluationRequest, SequenceIdentity, StepIdentity,
};
use runmat_analysis_fea::results::field::RecoveredFieldSelection;
use runmat_analysis_fea::results::observation::{
    resolve_observation_plan, ObservationArtifact, ObservationLimits,
};
use runmat_analysis_fea::topology::AnalysisProjection;
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    import_analysis_objects, import_analysis_output_set, import_analysis_partition,
    prepare_analysis_objects, prepare_analysis_program, prepare_partitioned_analysis_objects,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisPartitionImport, AnalysisWorkloadInput, AnalysisWorkloadRequest,
    ImportedRecoveredFields, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralObservationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_request: FieldRecoveryRequest,
    pub source_fields: ImportedRecoveredFields,
    pub mesh: PreparedAnalysisObjects,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub limits: ObservationLimits,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralObservation {
    pub request: ObservationEvaluationRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservations {
    pub artifact: ObservationArtifact,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_observation(
    source: &impl CacheImport,
    preparation: StructuralObservationPreparation,
) -> AnalysisBridgeResult<PreparedStructuralObservation> {
    preparation.source_host.validate()?;
    preparation.source_request.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::RecoverField
        || preparation.source_host.workload.outputs != vec![AnalysisNumericalObjectKind::Field]
        || preparation.source_fields.output.kind != AnalysisNumericalObjectKind::Field
        || preparation.mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation evaluation requires completed recovered fields, their mesh, and bounded chunks"
                .into(),
        ));
    }
    let request_input = unique_workload_input(
        &preparation.source_host,
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
    )?;
    let mesh_input =
        unique_workload_input(&preparation.source_host, AnalysisNumericalObjectKind::Mesh)?;
    if request_input.logical_identity != preparation.source_request.identity().digest()
        || preparation.source_host.workload.step != preparation.source_request.producer
        || mesh_input.logical_identity != preparation.mesh.manifest.logical_identity
        || mesh_input.payload_digest != preparation.mesh.manifest.payload_digest
        || preparation.source_request.mesh_digest != preparation.mesh.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "observation source differs from its field-recovery workload",
        ));
    }
    let expected_step =
        StepIdentity::from_solver_contract(preparation.sequence, &preparation.step)?;
    if expected_step != preparation.source_request.producer {
        return Err(AnalysisBridgeError::Identity(
            "observation procedure step differs from its producing step",
        ));
    }
    let mesh_artifact = SolverMeshArtifact::canonical_decode(&preparation.mesh.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let projection = AnalysisProjection::derive(&mesh_artifact)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if projection.projection_digest != preparation.source_request.projection_digest {
        return Err(AnalysisBridgeError::Identity(
            "observation projection differs from its field-recovery request",
        ));
    }
    let fields = &preparation.source_fields.manifest;
    if fields.identity.digest() != preparation.source_fields.output.logical_identity
        || fields.recovery_request != preparation.source_request.identity()
        || fields.source_result != preparation.source_request.source_result
    {
        return Err(AnalysisBridgeError::Identity(
            "recovered-field artifact differs from its source request",
        ));
    }
    let definitions = fields
        .fields
        .iter()
        .map(|field| field.definition.clone())
        .collect::<Vec<_>>();
    let plan = resolve_observation_plan(
        &preparation.step,
        &projection,
        &definitions,
        preparation.limits,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let required_definitions = plan
        .observations
        .iter()
        .map(|observation| observation.source_field_definition_identity)
        .collect::<BTreeSet<_>>();
    let selected_fields = fields
        .fields
        .iter()
        .filter(|entry| required_definitions.contains(&entry.definition.identity()))
        .map(|entry| entry.field)
        .collect::<Vec<_>>();
    let selection = RecoveredFieldSelection::new(fields.clone(), selected_fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let request = ObservationEvaluationRequest::new(
        preparation.source_request.producer,
        &selection,
        preparation.source_request.mesh_digest,
        plan,
    )?;
    let selection_bytes = selection
        .canonical_bytes()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mut selected_payloads = Vec::with_capacity(selection.fields.len());
    for field in &selection.fields {
        selected_payloads.push((
            *field,
            import_analysis_partition(
                source,
                AnalysisPartitionImport {
                    root: &preparation.source_fields.output.root,
                    expected_kind: AnalysisNumericalObjectKind::Field,
                    expected_logical_identity: preparation.source_fields.output.logical_identity,
                    expected_payload_digest: preparation.source_fields.output.payload_digest,
                    partition_identity: field.digest(),
                    access: &preparation.source_host.artifact_access,
                    limits: preparation.inventory_limits,
                },
            )?
            .payload(),
        ));
    }
    let mut field_partitions = Vec::with_capacity(selected_payloads.len() + 1);
    field_partitions.push(runmat_analysis_fea::execution::AnalysisPayloadPartition {
        logical_identity: selection.identity.digest(),
        payload: &selection_bytes,
    });
    field_partitions.extend(selected_payloads.iter().map(|(identity, payload)| {
        runmat_analysis_fea::execution::AnalysisPayloadPartition {
            logical_identity: identity.digest(),
            payload,
        }
    }));
    let selected_field_objects = prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::FieldSelection,
        selection.identity.digest(),
        &field_partitions,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
        request.identity().digest(),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (selected_field_objects, ValueRefKind::DriverObject),
        (preparation.mesh, ValueRefKind::DriverObject),
    ];
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
        preparation.source_host.workload.study,
        preparation.source_host.workload.step,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::EvaluateObservation,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::Observation],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.source_host.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralObservation {
        request,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_observations(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralObservation,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedObservations> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "observation evaluation must return exactly one observation root".into(),
        ));
    };
    let imported = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::Observation,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        imported.references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "observation result inventory differs from its canonical closure",
        ));
    }
    let artifact = ObservationArtifact::from_canonical_bytes(&imported.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if artifact.identity().digest() != output.logical_identity
        || artifact.evaluation_request != prepared.request.identity()
        || artifact.producer != prepared.request.producer
        || artifact.field_recovery_request != prepared.request.field_recovery_request
        || artifact.source_fields != prepared.request.source_fields
        || artifact.source_result != prepared.request.source_result
        || artifact.topology_digest != prepared.request.plan.topology_digest
        || artifact.plan_identity != prepared.request.plan.identity
    {
        return Err(AnalysisBridgeError::Identity(
            "observation artifact differs from its admitted evaluation request",
        ));
    }
    Ok(ImportedObservations {
        artifact,
        objects: imported,
    })
}

fn unique_workload_input(
    host: &AnalysisHostWorkload,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&AnalysisWorkloadInput> {
    let mut matches = host
        .workload
        .inputs
        .iter()
        .filter(|input| input.kind == kind);
    let input = matches.next().ok_or_else(|| {
        AnalysisBridgeError::Invalid(format!("analysis workload is missing its {kind:?} input"))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}
