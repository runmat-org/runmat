use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, AnalysisWorkloadIdentity,
    ObservationEvaluationRequest, ObservationPartitionEvaluationRequest,
    ObservationPartitionReductionRequest,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_partition_resource_plan, build_observation_reduction_resource_plan,
    execute_scoped_chunks, observation_partition_payload_bytes_upper_bound,
    observation_reduction_payload_bytes_upper_bound, FeaResourcePlan, LocalExecutionPlan,
    ObservationPartitionResourcePlanning, ObservationPartitionSourceWorkspace,
    ScopedLocalExecutionError,
};
use runmat_analysis_fea::results::field::{RecoveredFieldPartitionSet, RecoveredFieldSample};
use runmat_analysis_fea::results::observation::{
    classify_observation_partition_acceleration, resolved_observation_accepts_sample,
    structural_observation_partition_source_indices, ObservationAccelerationEligibility,
    ObservationEventAttribution, ObservationPartition, ObservationPartitionContribution,
    ObservationPartitionPlan, ObservationResult, SolidObservationSourceSelection,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, prepared_input_resource_facts, projected_output_resource_facts,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputRoot, AnalysisWorkloadInput, AnalysisWorkloadRequest, ObservationPreparation,
    PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

use super::observation_evaluation::{admit_observation, AdmittedObservation};
use super::observation_field_partitions::{
    import_required_field_partitions, RequiredFieldPartitionImport,
};

/// Authenticate and profile an observation partition's immutable input closure before
/// decoding typed source/catalog/field values. The worker still rederives the exact plan.
pub fn require_observation_partition_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    use runmat_analysis_fea::planning::resource::{FeaResourceDemand, FeaResourcePhase};
    plan.validate_identity()?;
    let FeaResourceDemand::StructuralObservationPartition(demand) = &plan.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "observation admission requires partition reservation",
        ));
    };
    let mut encoded = 0_u64;
    for input in inputs {
        runmat_execution_artifact::object::validate_inventory(&input.objects, limits)?;
        crate::portable_source_admission::authenticate_source_objects(input, input.manifest.kind)?;
        if input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan {
            if crate::decode_admitted_resource_plan(input, limits)? != *plan {
                return Err(AnalysisBridgeError::Identity(
                    "observation carried resource plan differs",
                ));
            }
        } else {
            encoded = encoded.checked_add(input.manifest.encoded_length).ok_or(
                AnalysisBridgeError::Identity("observation source wire overflow"),
            )?;
        }
    }
    let requests = inputs
        .iter()
        .filter(|i| {
            i.manifest.kind == AnalysisNumericalObjectKind::ObservationPartitionEvaluationRequest
        })
        .collect::<Vec<_>>();
    if requests.len() != 1
        || !matches!(plan.input.phase, FeaResourcePhase::StructuralObservationPartition(id) if id.digest() == requests[0].manifest.logical_identity)
        || encoded != demand.inputs.encoded_payload_bytes.0
        || encoded > demand.inputs.decoded_resident_bytes_admission_budget.0
    {
        return Err(AnalysisBridgeError::Identity(
            "observation request or raw decode budget differs",
        ));
    }
    let actual = crate::prepared_input_resource_facts_without_resource_plan(inputs)?;
    if actual != demand.inputs {
        return Err(AnalysisBridgeError::Identity(
            "observation actual decode profile differs",
        ));
    }
    Ok(())
}

pub struct PreparedObservationPartition {
    pub source_request: ObservationEvaluationRequest,
    pub request: ObservationPartitionEvaluationRequest,
    pub partition_plan: ObservationPartitionPlan,
    pub partition: ObservationPartition,
    pub source_fields: RecoveredFieldPartitionSet,
    execution_target: ObservationExecutionTarget,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ObservationExecutionTarget {
    Host,
    GovernedAccelerator {
        class: runmat_execution::resource::AcceleratorClass,
    },
}

pub struct PreparedObservationFanout {
    pub source_request: ObservationEvaluationRequest,
    /// The admitted source remains available when this sample selects no
    /// partition tasks. Child-task inputs do not own the fanout's identity.
    pub source_catalog: PreparedAnalysisObjects,
    pub partitions: Vec<PreparedObservationPartition>,
    /// A volume-mesh source's catalog with the plan of `source_request`, sealed at admission.
    pub volume_mesh:
        Option<runmat_analysis_fea::results::observation::VolumeMeshObservationAuthority>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedObservationPartitionContribution {
    pub contribution: ObservationPartitionContribution,
    pub objects: PreparedAnalysisObjects,
    pub output: AnalysisOutputRoot,
}

pub struct ObservationReductionPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_request: ObservationEvaluationRequest,
    pub partition_plan: ObservationPartitionPlan,
    pub contributions: Vec<ImportedObservationPartitionContribution>,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedObservationReduction {
    pub source_request: ObservationEvaluationRequest,
    pub request: ObservationPartitionReductionRequest,
    pub partition_plan: ObservationPartitionPlan,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub event_attribution: Option<ObservationEventAttribution>,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedReducedObservation {
    pub result: ObservationResult,
    pub objects: PreparedAnalysisObjects,
    pub(crate) producer: AnalysisWorkloadIdentity,
}

pub fn prepare_observation_fanout(
    source: &impl CacheImport,
    preparation: ObservationPreparation,
    sample: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
) -> AnalysisBridgeResult<PreparedObservationFanout> {
    let admitted = admit_observation(&preparation)?;
    if let Some(attribution) = event_attribution {
        attribution
            .validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if attribution.sample() != sample {
            return Err(AnalysisBridgeError::Identity(
                "observation fan-out event differs from its source sample",
            ));
        }
    }
    let solid = matches!(
        &admitted.workload,
        super::observation_source::AdmittedObservationWorkload::Thermal(_)
    ) || matches!(
        &preparation.source,
        crate::ObservationPublishedSource::VolumeMeshRecovery { .. }
    );
    let sampling = match &preparation.source {
        crate::ObservationPublishedSource::SampledRecovery { sampling, .. } => Some(sampling),
        _ => None,
    };
    // Every accepted (observation, partition) pair, in canonical plan order.
    let mut items = Vec::new();
    for observation in &admitted.request.plan.observations {
        if !resolved_observation_accepts_sample(
            &admitted.request.plan,
            observation,
            sample,
            event_attribution,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
        {
            continue;
        }
        let plan = ObservationPartitionPlan::derive(
            &admitted.request.plan,
            &observation.spec.observation_id,
            admitted.workload.observation_execution(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let field_entry = admitted
            .selection
            .fields()
            .iter()
            .find(|entry| {
                entry.definition.identity() == observation.source_field_definition_identity
                    && entry.sample == sample
                    && admitted
                        .selection
                        .fields()
                        .iter()
                        .any(|field| field.field == entry.field)
            })
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "observation fan-out source sample is absent from recovered fields".into(),
                )
            })?;
        let plan = std::sync::Arc::new(plan);
        for ordinal in 0..plan.partitions.len() {
            items.push((plan.clone(), ordinal, field_entry));
        }
    }
    let local_execution = preparation.local_execution;
    // A solid or volume plan selects every partition from one projection of the exact mesh and
    // one observation topology of the plan, both derived once per fan-out.
    let volume_projection = if !items.is_empty() && sampling.is_none() && solid {
        let mesh = admitted.mesh.as_ref().ok_or(AnalysisBridgeError::Identity(
            "solid-projection observation requires exact mesh",
        ))?;
        Some(
            runmat_analysis_fea::topology::AnalysisProjection::derive(mesh)
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )
    } else {
        None
    };
    let solid_selection = volume_projection
        .as_ref()
        .map(|projection| SolidObservationSourceSelection::new(&admitted.request.plan, projection))
        .transpose()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let required_indices = fan_out(local_execution, &items, |(plan, ordinal, _)| {
        let partition = &plan.partitions[*ordinal];
        if let Some(sampling) = sampling {
            let mesh = admitted.mesh.as_ref().ok_or(AnalysisBridgeError::Identity(
                "sampled observation mesh absent",
            ))?;
            runmat_analysis_fea::results::observation::sampled_observation_partition_source_indices(
                &admitted.request.plan,
                plan,
                partition,
                sampling,
                mesh,
            )
        } else if solid {
            solid_selection
                .as_ref()
                .ok_or(AnalysisBridgeError::Identity(
                    "solid-projection observation requires exact mesh",
                ))?
                .partition_source_indices(plan, partition)
        } else {
            structural_observation_partition_source_indices(
                &admitted.request.plan,
                plan,
                partition,
                admitted.workload.model()?,
                admitted.mesh.as_ref(),
            )
        }
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
    })?;
    drop(solid_selection);
    // Field partitions are read from the cache in canonical order on this thread.
    let mut imported = Vec::with_capacity(items.len());
    for ((plan, ordinal, field_entry), required_indices) in items.into_iter().zip(required_indices)
    {
        let source_fields = import_required_field_partitions(
            source,
            RequiredFieldPartitionImport {
                source_kind: preparation.source.output().kind,
                root: &preparation.source.output().root,
                source_logical_identity: preparation.source.output().logical_identity,
                source_payload_digest: preparation.source.output().payload_digest,
                access: preparation.source.access(&preparation.source_host),
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
        imported.push((plan, ordinal, source_fields));
    }
    let source_request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
        admitted.request.identity().digest(),
        &admitted.request.canonical_bytes()?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let source_workspace = observation_partition_source_workspace(&preparation, &admitted)?;
    let prepared = fan_out(
        local_execution,
        &imported,
        |(plan, ordinal, source_fields)| {
            prepare_partition_workload(
                &preparation,
                &admitted,
                &source_request_objects,
                PartitionWork {
                    partition_plan: plan.as_ref().clone(),
                    partition: plan.partitions[*ordinal].clone(),
                    source_fields: source_fields.clone(),
                    event_attribution,
                    source_workspace,
                },
            )
        },
    )?;
    Ok(PreparedObservationFanout {
        source_request: admitted.request,
        source_catalog: admitted.field_catalog,
        partitions: prepared,
        volume_mesh: admitted.volume_mesh,
    })
}

/// Maps independent fan-out items on the admitted local CPU plan. Each item is computed exactly
/// as on one thread; results return in canonical item order, so the fan-out is identical for
/// every plan.
fn fan_out<I, T, F>(
    local_execution: LocalExecutionPlan,
    items: &[I],
    work: F,
) -> AnalysisBridgeResult<Vec<T>>
where
    I: Sync,
    T: Send,
    F: Fn(&I) -> AnalysisBridgeResult<T> + Sync,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let chunks = execute_scoped_chunks(local_execution, items.len() as u64, |chunk| {
        items[chunk.start() as usize..chunk.end_exclusive() as usize]
            .iter()
            .map(&work)
            .collect::<AnalysisBridgeResult<Vec<_>>>()
    })
    .map_err(|error| match error {
        ScopedLocalExecutionError::Work { source, .. } => source,
        other => AnalysisBridgeError::Invalid(other.to_string()),
    })?;
    Ok(chunks.into_iter().flatten().collect())
}

pub fn import_observation_partition_contribution(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedObservationPartition,
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
    let mut expected =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.push(output_set.root.clone());
    if expected != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "observation contribution inventory differs from canonical closure",
        ));
    }
    let contribution = ObservationPartitionContribution::from_canonical_bytes(&objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if contribution.identity.digest() != output.logical_identity
        || contribution.partition_plan != prepared.partition_plan.identity
        || contribution.partition != prepared.partition.identity
        || contribution.partition_ordinal != prepared.partition.ordinal
        || contribution.source_field != prepared.source_fields.source_field
        || contribution.event_attribution != prepared.request.event_attribution
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
    match (
        &prepared.execution_target,
        contribution.acceleration.as_ref(),
    ) {
        (ObservationExecutionTarget::Host, None) => {}
        (ObservationExecutionTarget::GovernedAccelerator { class }, Some(evidence)) => {
            let [allocation] = prepared.resource_plan.request.accelerators.as_slice() else {
                return Err(AnalysisBridgeError::Identity(
                    "accelerated observation contribution lacks one exact allocation",
                ));
            };
            validate_acceleration_binding(
                class,
                prepared.resource_plan.identity,
                allocation,
                evidence.resource_plan,
                &evidence.allocation,
            )?;
            evidence
                .validate_for_allocation(allocation)
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        }
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "observation acceleration evidence differs from prepared placement",
            ))
        }
    }
    Ok(ImportedObservationPartitionContribution {
        contribution,
        objects,
        output: output.clone(),
    })
}

fn validate_acceleration_binding(
    prepared_class: &runmat_execution::resource::AcceleratorClass,
    prepared_plan: runmat_analysis_fea::planning::resource::FeaResourcePlanIdentity,
    prepared_allocation: &AcceleratorRequest,
    evidence_plan: runmat_analysis_fea::planning::resource::FeaResourcePlanIdentity,
    evidence_allocation: &AcceleratorRequest,
) -> AnalysisBridgeResult<()> {
    if &prepared_allocation.class != prepared_class
        || prepared_plan != evidence_plan
        || prepared_allocation != evidence_allocation
    {
        return Err(AnalysisBridgeError::Identity(
            "observation contribution plan or allocation differs from prepared placement",
        ));
    }
    Ok(())
}

pub fn prepare_observation_reduction(
    preparation: ObservationReductionPreparation,
) -> AnalysisBridgeResult<PreparedObservationReduction> {
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
    let event_attribution = contributions[0].event_attribution;
    let request = ObservationPartitionReductionRequest::new(
        &preparation.source_request,
        &preparation.partition_plan,
        &contributions,
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
    let input_facts = prepared_input_resource_facts(&inputs)?;
    let output_payload_upper_bound =
        observation_reduction_payload_bytes_upper_bound(&input_facts, &contributions)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = build_observation_reduction_resource_plan(
        &preparation.source_request,
        &request,
        &preparation.partition_plan,
        &contributions,
        input_facts,
        projected_output_resource_facts(output_payload_upper_bound)?,
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
        AnalysisWorkKind::ReduceObservationPartitions,
        &inputs,
        vec![AnalysisNumericalObjectKind::ObservationResult],
        BTreeSet::new(),
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision.clone(), &inputs)?;
    Ok(PreparedObservationReduction {
        source_request: preparation.source_request,
        request,
        partition_plan: preparation.partition_plan,
        resource_plan,
        resource_plan_objects,
        event_attribution,
        host,
        program,
        inputs,
    })
}

pub fn import_reduced_observation(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedObservationReduction,
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
    let mut expected =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.push(output_set.root.clone());
    if expected != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "observation reduction inventory differs from canonical closure",
        ));
    }
    let result = ObservationResult::from_canonical_bytes(&objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let contributions = prepared
        .inputs
        .iter()
        .filter(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::ObservationPartitionContribution
        })
        .map(|input| {
            ObservationPartitionContribution::from_canonical_bytes(&input.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let expected = runmat_analysis_fea::results::observation::reduce_observation_partitions(
        &prepared.source_request.plan,
        &prepared.partition_plan,
        contributions,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if result.logical_identity.digest() != output.logical_identity
        || result.plan_identity != prepared.source_request.plan.identity
        || result.topology_digest != prepared.source_request.plan.topology_digest
        || result.partition_evidence.partition_plan != prepared.partition_plan.identity
        || result.partition_evidence.event_attribution != prepared.event_attribution
        || result != expected
    {
        return Err(AnalysisBridgeError::Identity(
            "reduced observation differs from its admitted request",
        ));
    }
    Ok(ImportedReducedObservation {
        result,
        objects,
        producer: prepared.host.workload.identity,
    })
}

/// Worker storage of the partition's source, read by the same law the worker re-derives.
fn observation_partition_source_workspace(
    preparation: &ObservationPreparation,
    admitted: &AdmittedObservation,
) -> AnalysisBridgeResult<ObservationPartitionSourceWorkspace> {
    match (
        &admitted.workload,
        &preparation.source,
        admitted.mesh.as_ref(),
    ) {
        (
            super::observation_source::AdmittedObservationWorkload::Thermal(workload),
            crate::ObservationPublishedSource::Thermal { manifest, .. },
            Some(mesh),
        ) => ObservationPartitionSourceWorkspace::thermal(
            workload,
            manifest,
            &admitted.request.plan,
            mesh,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string())),
        (super::observation_source::AdmittedObservationWorkload::Thermal(_), _, _) => Err(
            AnalysisBridgeError::Identity("thermal observation requires its manifest and mesh"),
        ),
        _ => Ok(ObservationPartitionSourceWorkspace::InputEnvelope),
    }
}

struct PartitionWork {
    partition_plan: ObservationPartitionPlan,
    partition: ObservationPartition,
    source_fields: RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
    source_workspace: ObservationPartitionSourceWorkspace,
}

fn prepare_partition_workload(
    preparation: &ObservationPreparation,
    admitted: &AdmittedObservation,
    source_request_objects: &PreparedAnalysisObjects,
    work: PartitionWork,
) -> AnalysisBridgeResult<PreparedObservationPartition> {
    let PartitionWork {
        partition_plan,
        partition,
        source_fields,
        event_attribution,
        source_workspace,
    } = work;
    let execution_target = observation_execution_target(
        &preparation.source_host,
        admitted,
        &partition_plan,
        &partition,
        &source_fields,
        event_attribution,
    )?;
    let request = ObservationPartitionEvaluationRequest::new(
        &admitted.request,
        &partition_plan,
        &partition,
        &source_fields,
        event_attribution,
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
        source_request_objects.clone(),
        plan_objects,
        field_objects,
        admitted.field_catalog.clone(),
        preparation.source_workload.objects().clone(),
    ];
    if let Some(mesh) = &preparation.mesh {
        inputs.push(mesh.clone());
    }
    let accelerator = match &execution_target {
        ObservationExecutionTarget::Host => None,
        ObservationExecutionTarget::GovernedAccelerator { class } => Some(
            admitted
                .accelerator_requests
                .iter()
                .find(|allocation| &allocation.class == class)
                .ok_or(AnalysisBridgeError::Identity(
                    "observation placement lacks its admitted accelerator allocation",
                ))?,
        ),
    };
    let input_facts = prepared_input_resource_facts(&inputs)?;
    let output_payload_upper_bound = observation_partition_payload_bytes_upper_bound(
        &admitted.request,
        &request,
        &partition_plan,
        &partition,
        &source_fields,
        accelerator.is_some(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan =
        build_observation_partition_resource_plan(ObservationPartitionResourcePlanning {
            source: &admitted.request,
            request: &request,
            partition_plan: &partition_plan,
            partition: &partition,
            source_fields: &source_fields,
            inputs: input_facts,
            output: projected_output_resource_facts(output_payload_upper_bound)?,
            output_set: crate::projected_output_set_resource_facts(),
            accelerator,
            source_workspace,
        })
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
        AnalysisWorkKind::EvaluateObservationPartition,
        &inputs,
        vec![AnalysisNumericalObjectKind::ObservationPartitionContribution],
        match &execution_target {
            ObservationExecutionTarget::Host => BTreeSet::new(),
            ObservationExecutionTarget::GovernedAccelerator { class } => {
                BTreeSet::from([AnalysisCapabilityRequirement::Accelerator {
                    class: class.clone(),
                }])
            }
        },
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision.clone(), &inputs)?;
    Ok(PreparedObservationPartition {
        source_request: admitted.request.clone(),
        request,
        partition_plan,
        partition,
        source_fields,
        execution_target,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs,
    })
}

fn observation_execution_target(
    source_host: &AnalysisHostWorkload,
    admitted: &AdmittedObservation,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    source_fields: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
) -> AnalysisBridgeResult<ObservationExecutionTarget> {
    let eligibility = classify_observation_partition_acceleration(
        &admitted.request.plan,
        partition_plan,
        partition,
        source_fields,
        event_attribution,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    select_observation_execution_target(
        admitted.workload.backend(),
        eligibility,
        &admitted.accelerator_requests,
        &source_host.workload.required_capabilities,
    )
}

fn select_observation_execution_target(
    backend: ComputeBackend,
    eligibility: ObservationAccelerationEligibility,
    accelerators: &[AcceleratorRequest],
    capabilities: &BTreeSet<AnalysisCapabilityRequirement>,
) -> AnalysisBridgeResult<ObservationExecutionTarget> {
    if backend != ComputeBackend::Gpu
        || eligibility != ObservationAccelerationEligibility::GovernedExactF64Extrema
    {
        return Ok(ObservationExecutionTarget::Host);
    }
    let candidates = accelerators
        .iter()
        .filter(|accelerator| {
            capabilities.contains(&AnalysisCapabilityRequirement::Accelerator {
                class: accelerator.class.clone(),
            })
        })
        .map(|accelerator| accelerator.class.clone())
        .collect::<BTreeSet<_>>();
    match candidates.len() {
        0 => Ok(ObservationExecutionTarget::Host),
        1 => Ok(ObservationExecutionTarget::GovernedAccelerator {
            class: candidates.into_iter().next().expect("one candidate"),
        }),
        _ => Err(AnalysisBridgeError::Invalid(
            "governed observation acceleration has multiple eligible accelerator classes".into(),
        )),
    }
}

pub(crate) fn prepare_host_workload(
    source_host: &AnalysisHostWorkload,
    kind: AnalysisWorkKind,
    inputs: &[PreparedAnalysisObjects],
    outputs: Vec<AnalysisNumericalObjectKind>,
    additional_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    output_policy: crate::AnalysisOutputPolicy,
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
    let mut required_capabilities = additional_capabilities;
    required_capabilities.insert(AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    });
    let workload = AnalysisWorkloadRequest::new(
        source_host.workload.study,
        source_host.workload.step,
        source_host.workload.execution_plan,
        kind,
        workload_inputs,
        outputs,
        required_capabilities,
    )?;
    AnalysisHostWorkload::new(workload, source_host.artifact_access.clone(), output_policy)
}

#[cfg(test)]
mod placement_tests {
    use std::collections::BTreeSet;

    use runmat_analysis_fea::results::observation::ObservationAccelerationEligibility;
    use runmat_analysis_fea::ComputeBackend;
    use runmat_execution::{resource::AcceleratorRequest, Digest};

    use crate::AnalysisCapabilityRequirement;

    use super::{select_observation_execution_target, validate_acceleration_binding};

    fn accelerator(class: &str) -> AcceleratorRequest {
        AcceleratorRequest {
            class: runmat_execution::resource::AcceleratorClass::new(class).unwrap(),
            count: 1,
            minimum_allocation_bytes: 1024,
            provider: None,
            required_features: BTreeSet::new(),
        }
    }

    fn capability(class: &str) -> AnalysisCapabilityRequirement {
        AnalysisCapabilityRequirement::Accelerator {
            class: runmat_execution::resource::AcceleratorClass::new(class).unwrap(),
        }
    }

    #[test]
    fn governed_placement_requires_gpu_eligibility_allocation_and_capability() {
        let eligible = ObservationAccelerationEligibility::GovernedExactF64Extrema;
        let allocations = vec![accelerator("gpu-a")];
        let capabilities = BTreeSet::from([capability("gpu-a")]);
        assert_eq!(
            select_observation_execution_target(
                ComputeBackend::Gpu,
                eligible,
                &allocations,
                &capabilities,
            )
            .unwrap(),
            super::ObservationExecutionTarget::GovernedAccelerator {
                class: runmat_execution::resource::AcceleratorClass::new("gpu-a").unwrap(),
            }
        );
        for (backend, eligibility, allocated, capable) in [
            (ComputeBackend::Cpu, eligible, true, true),
            (
                ComputeBackend::Gpu,
                ObservationAccelerationEligibility::Host,
                true,
                true,
            ),
            (ComputeBackend::Gpu, eligible, false, true),
            (ComputeBackend::Gpu, eligible, true, false),
        ] {
            let allocations = allocated
                .then(|| accelerator("gpu-a"))
                .into_iter()
                .collect::<Vec<_>>();
            let capabilities = capable
                .then(|| capability("gpu-a"))
                .into_iter()
                .collect::<BTreeSet<_>>();
            assert_eq!(
                select_observation_execution_target(
                    backend,
                    eligibility,
                    &allocations,
                    &capabilities,
                )
                .unwrap(),
                super::ObservationExecutionTarget::Host
            );
        }
    }

    #[test]
    fn governed_placement_rejects_ambiguous_accelerator_classes() {
        assert!(select_observation_execution_target(
            ComputeBackend::Gpu,
            ObservationAccelerationEligibility::GovernedExactF64Extrema,
            &[accelerator("gpu-a"), accelerator("gpu-b")],
            &BTreeSet::from([capability("gpu-a"), capability("gpu-b")]),
        )
        .is_err());
    }

    #[test]
    fn homogeneous_acceleration_allocations_cannot_swap_phase_plans() {
        let allocation = accelerator("gpu-a");
        let first =
            serde_json::from_value(serde_json::to_value(Digest::sha256("first")).unwrap()).unwrap();
        let second =
            serde_json::from_value(serde_json::to_value(Digest::sha256("second")).unwrap())
                .unwrap();
        validate_acceleration_binding(
            &runmat_execution::resource::AcceleratorClass::new("gpu-a").unwrap(),
            first,
            &allocation,
            first,
            &allocation,
        )
        .unwrap();
        assert!(validate_acceleration_binding(
            &allocation.class,
            first,
            &allocation,
            second,
            &allocation,
        )
        .is_err());
    }
}
