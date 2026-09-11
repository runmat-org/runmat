use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ObservationEvaluationRequest,
    ObservationPartitionEvaluationRequest, ObservationPartitionReductionRequest,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_analysis_fea::results::field::{RecoveredFieldPartitionSet, RecoveredFieldSample};
use runmat_analysis_fea::results::observation::{
    structural_observation_partition_source_indices, ObservationPartition,
    ObservationPartitionContribution, ObservationPartitionPlan, ObservationResult,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisOutputRoot, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, PreparedAnalysisProgram,
    StructuralObservationPreparation, ANALYSIS_HOST_ABI,
};

use super::observation_evaluation::{admit_structural_observation, AdmittedStructuralObservation};
use super::observation_field_partitions::{
    import_required_field_partitions, RequiredFieldPartitionImport,
};

pub struct PreparedStructuralObservationPartition {
    pub source_request: ObservationEvaluationRequest,
    pub request: ObservationPartitionEvaluationRequest,
    pub partition_plan: ObservationPartitionPlan,
    pub partition: ObservationPartition,
    pub source_fields: RecoveredFieldPartitionSet,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

pub struct PreparedStructuralObservationFanout {
    pub source_request: ObservationEvaluationRequest,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub partitions: Vec<PreparedStructuralObservationPartition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservationPartitionContribution {
    pub contribution: ObservationPartitionContribution,
    pub objects: PreparedAnalysisObjects,
    pub output: AnalysisOutputRoot,
}

pub struct StructuralObservationReductionPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_request: ObservationEvaluationRequest,
    pub partition_plan: ObservationPartitionPlan,
    pub contributions: Vec<ImportedObservationPartitionContribution>,
    pub expected_resource_plan: FeaResourcePlan,
    pub resource_plan: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralObservationReduction {
    pub source_request: ObservationEvaluationRequest,
    pub request: ObservationPartitionReductionRequest,
    pub partition_plan: ObservationPartitionPlan,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedReducedObservation {
    pub result: ObservationResult,
    pub objects: PreparedAnalysisObjects,
    pub output: AnalysisOutputRoot,
}

pub fn prepare_structural_observation_fanout(
    source: &impl CacheImport,
    preparation: StructuralObservationPreparation,
    sample: RecoveredFieldSample,
) -> AnalysisBridgeResult<PreparedStructuralObservationFanout> {
    let admitted = admit_structural_observation(&preparation)?;
    let mut prepared = Vec::new();
    for observation in &admitted.request.plan.observations {
        let plan = ObservationPartitionPlan::derive(
            &admitted.request.plan,
            &observation.spec.observation_id,
            admitted.workload.observation_execution,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let field_entry = preparation
            .source_fields
            .manifest
            .fields
            .iter()
            .find(|entry| {
                entry.definition.identity() == observation.source_field_definition_identity
                    && entry.sample == sample
                    && admitted.selection.fields.contains(&entry.field)
            })
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "observation fan-out source sample is absent from recovered fields".into(),
                )
            })?;
        for partition in &plan.partitions {
            let required_indices = structural_observation_partition_source_indices(
                &admitted.request.plan,
                &plan,
                partition,
                &admitted.workload.model,
                admitted.mesh.as_ref(),
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            let source_fields = import_required_field_partitions(
                source,
                RequiredFieldPartitionImport {
                    root: &preparation.source_fields.output.root,
                    source_logical_identity: preparation.source_fields.output.logical_identity,
                    source_payload_digest: preparation.source_fields.output.payload_digest,
                    access: &preparation.source_host.artifact_access,
                    limits: preparation.inventory_limits,
                    entry: field_entry,
                    required_indices: &required_indices,
                },
            )?;
            if required_indices
                .iter()
                .any(|index| source_fields.value(*index).is_none())
            {
                return Err(AnalysisBridgeError::Identity(
                    "selected storage partitions do not cover the observation partition",
                ));
            }
            prepared.push(prepare_partition_workload(
                &preparation,
                &admitted,
                plan.clone(),
                partition.clone(),
                source_fields,
            )?);
        }
    }
    if prepared.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "observation fan-out produced no partition tasks".into(),
        ));
    }
    Ok(PreparedStructuralObservationFanout {
        source_request: admitted.request,
        resource_plan: admitted.resource_plan,
        resource_plan_objects: preparation.resource_plan,
        partitions: prepared,
    })
}

pub fn import_observation_partition_contribution(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralObservationPartition,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedObservationPartitionContribution> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "observation partition evaluation must return one contribution".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::ObservationPartitionContribution,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let contribution = ObservationPartitionContribution::from_canonical_bytes(&objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if contribution.identity.digest() != output.logical_identity
        || contribution.partition_plan != prepared.partition_plan.identity
        || contribution.partition != prepared.partition.identity
        || contribution.partition_ordinal != prepared.partition.ordinal
        || contribution.source_field != prepared.source_fields.source_field
        || contribution.source_partitions
            != prepared
                .source_fields
                .partitions
                .iter()
                .map(|partition| partition.identity)
                .collect::<Vec<_>>()
    {
        return Err(AnalysisBridgeError::Identity(
            "observation contribution differs from its admitted partition",
        ));
    }
    Ok(ImportedObservationPartitionContribution {
        contribution,
        objects,
        output: output.clone(),
    })
}

pub fn prepare_structural_observation_reduction(
    preparation: StructuralObservationReductionPreparation,
) -> AnalysisBridgeResult<PreparedStructuralObservationReduction> {
    preparation.source_host.validate()?;
    preparation.source_request.validate()?;
    preparation
        .partition_plan
        .validate_for(&preparation.source_request.plan)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if preparation.chunk_bytes == 0 || preparation.contributions.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "observation reduction requires contributions and bounded chunks".into(),
        ));
    }
    let contributions = preparation
        .contributions
        .iter()
        .map(|imported| imported.contribution.clone())
        .collect::<Vec<_>>();
    let request = ObservationPartitionReductionRequest::new(
        &preparation.source_request,
        &preparation.partition_plan,
        &contributions,
        &preparation.expected_resource_plan,
    )?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationPartitionReductionRequest,
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
    let plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationPartitionPlan,
        preparation.partition_plan.identity.digest(),
        &preparation
            .partition_plan
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = vec![request_objects, source_request_objects, plan_objects];
    inputs.extend(
        preparation
            .contributions
            .into_iter()
            .map(|imported| imported.objects),
    );
    inputs.push(preparation.resource_plan.clone());
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&preparation.resource_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan != preparation.expected_resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "observation reduction received a substituted resource plan",
        ));
    }
    let host = prepare_host_workload(
        &preparation.source_host,
        AnalysisWorkKind::ReduceObservationPartitions,
        &inputs,
        vec![AnalysisNumericalObjectKind::ObservationResult],
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision.clone(), &inputs)?;
    Ok(PreparedStructuralObservationReduction {
        source_request: preparation.source_request,
        request,
        partition_plan: preparation.partition_plan,
        resource_plan,
        resource_plan_objects: preparation.resource_plan,
        host,
        program,
        inputs,
    })
}

pub fn import_reduced_observation(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralObservationReduction,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedReducedObservation> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "observation reduction must return one result".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::ObservationResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = ObservationResult::from_canonical_bytes(&objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if result.logical_identity.digest() != output.logical_identity
        || result.plan_identity != prepared.source_request.plan.identity
        || result.topology_digest != prepared.source_request.plan.topology_digest
        || result.partition_evidence.partition_plan != prepared.partition_plan.identity
    {
        return Err(AnalysisBridgeError::Identity(
            "reduced observation differs from its admitted request",
        ));
    }
    Ok(ImportedReducedObservation {
        result,
        objects,
        output: output.clone(),
    })
}

fn prepare_partition_workload(
    preparation: &StructuralObservationPreparation,
    admitted: &AdmittedStructuralObservation,
    partition_plan: ObservationPartitionPlan,
    partition: ObservationPartition,
    source_fields: RecoveredFieldPartitionSet,
) -> AnalysisBridgeResult<PreparedStructuralObservationPartition> {
    let request = ObservationPartitionEvaluationRequest::new(
        &admitted.request,
        &partition_plan,
        &partition,
        &source_fields,
        &admitted.resource_plan,
    )?;
    let source_request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
        admitted.request.identity().digest(),
        &admitted.request.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationPartitionEvaluationRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let plan_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationPartitionPlan,
        partition_plan.identity.digest(),
        &partition_plan
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let field_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldPartition,
        source_fields.identity.digest(),
        &source_fields
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut inputs = vec![
        request_objects,
        source_request_objects,
        plan_objects,
        field_objects,
        preparation.source_invocation.clone(),
        preparation.resource_plan.clone(),
    ];
    if let Some(mesh) = &preparation.mesh {
        inputs.push(mesh.clone());
    }
    let host = prepare_host_workload(
        &preparation.source_host,
        AnalysisWorkKind::EvaluateObservationPartition,
        &inputs,
        vec![AnalysisNumericalObjectKind::ObservationPartitionContribution],
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision.clone(), &inputs)?;
    Ok(PreparedStructuralObservationPartition {
        source_request: admitted.request.clone(),
        request,
        partition_plan,
        partition,
        source_fields,
        host,
        program,
        inputs,
    })
}

pub(crate) fn prepare_host_workload(
    source_host: &AnalysisHostWorkload,
    kind: AnalysisWorkKind,
    inputs: &[PreparedAnalysisObjects],
    outputs: Vec<AnalysisNumericalObjectKind>,
) -> AnalysisBridgeResult<AnalysisHostWorkload> {
    let workload_inputs = inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        })
        .collect();
    let workload = AnalysisWorkloadRequest::new(
        source_host.workload.study,
        source_host.workload.step,
        source_host.workload.execution_plan,
        kind,
        workload_inputs,
        outputs,
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    AnalysisHostWorkload::new(workload, source_host.artifact_access.clone())
}
