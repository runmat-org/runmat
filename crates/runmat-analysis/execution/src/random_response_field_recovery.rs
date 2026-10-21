//! Source-bound recovery of a random-response step's RMS fields for authored observations.
//! The task copies the original assembled result's fields; it never re-solves.
use crate::*;
use runmat_analysis_fea::{
    execution::{
        random_response_statistical_fields, statistical_field_request, AnalysisNumericalObjectKind,
        AnalysisTransportProfiler, AnalysisWorkKind, ExecutionPlanIdentity, FieldRecoveryRequest,
        NumericalPlanIdentity, StructuralRandomResponseSolverResult,
        StructuralRandomResponseWorkload,
    },
    planning::resource::{
        build_random_response_field_recovery_resource_plan, random_response_field_recovery_shape,
        ByteCount, FeaResourcePlan, PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct RandomResponseFieldRecoveryPreparation {
    /// The completed random-response assembly that produced `source_result`.
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
}

pub struct PreparedRandomResponseFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
}

pub fn prepare_random_response_field_recovery(
    p: RandomResponseFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedRandomResponseFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::AssembleStructuralRandomResponseResult {
        return Err(invalid(
            "random-response field recovery requires its completed assembly",
        ));
    }
    for (kind, objects) in [
        (
            AnalysisNumericalObjectKind::StructuralRandomResponseWorkload,
            &p.workload,
        ),
        (AnalysisNumericalObjectKind::Mesh, &p.mesh),
    ] {
        let source = p
            .source_host
            .workload
            .inputs
            .iter()
            .filter(|input| input.kind == kind)
            .collect::<Vec<_>>();
        if source.len() != 1
            || objects.manifest.kind != kind
            || source[0].logical_identity != objects.manifest.logical_identity
            || source[0].payload_digest != objects.manifest.payload_digest
            || source[0].manifest_digest != objects.root.digest
            || source[0].manifest_encoded_length != objects.root.encoded_length
        {
            return Err(invalid(
                "random-response recovery original source closure differs",
            ));
        }
    }
    let source_identity = runmat_analysis_fea::execution::SolverResultIdentity::new(
        p.source_host.workload.identity,
        p.source_result.manifest.payload_digest,
    )?;
    if source_identity.digest() != p.source_result.manifest.logical_identity {
        return Err(invalid(
            "random-response recovery source differs from its assembly-bound result",
        ));
    }
    let request = random_response_recovery_request(
        p.source_host.workload.step,
        source_identity,
        &p.workload,
        &p.mesh,
        &p.source_result,
    )?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?;
    let mut inputs = vec![request_objects, p.workload, p.mesh, p.source_result];
    let resource_plan = build_prepared_random_response_field_recovery_resource_plan(
        &request,
        &inputs,
        p.output_policy,
    )?;
    inputs.push(prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?);
    let execution = execution_plan(&request, resource_plan.identity.digest())?;
    let request_host = AnalysisWorkloadRequest::new(
        p.source_host.workload.study,
        p.source_host.workload.step,
        execution,
        AnalysisWorkKind::RecoverField,
        host_inputs(&inputs),
        vec![AnalysisNumericalObjectKind::Field],
        p.source_host.workload.required_capabilities.clone(),
    )?;
    let host =
        AnalysisHostWorkload::new(request_host, p.source_host.artifact_access, p.output_policy)?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedRandomResponseFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
    })
}

/// The one recovery request of an exact random-response result on its canonical mesh.
pub fn random_response_recovery_request(
    producer: runmat_analysis_fea::execution::StepIdentity,
    identity: runmat_analysis_fea::execution::SolverResultIdentity,
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<FieldRecoveryRequest> {
    if identity.digest() != source.manifest.logical_identity {
        return Err(invalid("random-response recovery source identity differs"));
    }
    let (_, mesh, result) = decode_random_response_recovery_source(workload, mesh, source)?;
    statistical_field_request(
        producer,
        identity,
        &mesh,
        &random_response_statistical_fields(result.numerical()),
    )
    .map_err(Into::into)
}

/// Decode and cross-validate the immutable recovery sources.
pub fn decode_random_response_recovery_source(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<(
    StructuralRandomResponseWorkload,
    SolverMeshArtifact,
    StructuralRandomResponseSolverResult,
)> {
    let w = StructuralRandomResponseWorkload::from_canonical_bytes(&workload.payload())?;
    let m = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    let r = StructuralRandomResponseSolverResult::from_canonical_bytes(&source.payload())?;
    if workload.manifest.kind != AnalysisNumericalObjectKind::StructuralRandomResponseWorkload
        || mesh.manifest.logical_identity != Digest::from_bytes(*m.canonical_digest.bytes())
        || source.manifest.kind != AnalysisNumericalObjectKind::SolverResult
        || r.numerical().fields().iter().any(|field| {
            field.shape().first().is_none_or(|count| {
                *count != m.topology.nodes.len() && *count != m.topology.volume_elements.len()
            })
        })
    {
        return Err(invalid("random-response recovery source identity differs"));
    }
    Ok((w, m, r))
}

fn execution_plan(
    request: &FieldRecoveryRequest,
    resource_plan: Digest,
) -> AnalysisBridgeResult<ExecutionPlanIdentity> {
    let source = request
        .source
        .solver_result()
        .ok_or_else(|| invalid("random-response recovery requires its solver result"))?;
    let numerical = NumericalPlanIdentity::from_statistical_field_recovery(
        source,
        request.topology.discretization_digest(),
    )?;
    Ok(ExecutionPlanIdentity::new(numerical, resource_plan)?)
}

fn host_inputs(inputs: &[PreparedAnalysisObjects]) -> Vec<AnalysisWorkloadInput> {
    inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: if input.manifest.kind == AnalysisNumericalObjectKind::SolverResult {
                ValueRefKind::ResultObject
            } else {
                ValueRefKind::DriverObject
            },
        })
        .collect()
}

/// Metadata-only rederivation used by preparation, the worker and import alike.
pub fn build_prepared_random_response_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let numerical = inputs
        .iter()
        .filter(|i| i.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    let kinds = [
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        AnalysisNumericalObjectKind::StructuralRandomResponseWorkload,
        AnalysisNumericalObjectKind::Mesh,
        AnalysisNumericalObjectKind::SolverResult,
    ];
    if numerical.len() != kinds.len()
        || inputs.len() - numerical.len() > 1
        || kinds.iter().any(|kind| {
            numerical
                .iter()
                .filter(|i| i.manifest.kind == *kind)
                .count()
                != 1
        })
    {
        return Err(invalid(
            "random-response recovery immutable input inventory differs",
        ));
    }
    let unique = |kind| {
        numerical
            .iter()
            .copied()
            .find(|i| i.manifest.kind == kind)
            .ok_or_else(|| invalid("missing random-response recovery input"))
    };
    if unique(AnalysisNumericalObjectKind::FieldRecoveryRequest)?
        .manifest
        .logical_identity
        != request.identity().digest()
    {
        return Err(invalid("random-response recovery request differs"));
    }
    let (partitions, _) = random_response_field_recovery_shape(request)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    // Decoding the three sources and validating their shapes is the only source work.
    let source_validation = numerical
        .iter()
        .map(|input| input.manifest.encoded_length)
        .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or_else(|| invalid("random-response recovery source bytes overflow"))?;
    Ok(build_random_response_field_recovery_resource_plan(
        request,
        ByteCount(source_validation.max(1)),
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_transport_output_resource_facts(&profile)?,
            output_set: projected_output_set_resource_facts(),
        },
    )?)
}

pub fn import_random_response_recovered_fields(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    prepared: &PreparedRandomResponseFieldRecovery,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
        || host_inputs(&prepared.inputs) != prepared.host.workload.inputs
    {
        return Err(invalid(
            "random-response field import requires its exact recovery task",
        ));
    }
    let expected = build_prepared_random_response_field_recovery_resource_plan(
        &prepared.request,
        &prepared.inputs,
        prepared.output_policy,
    )?;
    if expected != prepared.resource_plan
        || prepared.host.workload.execution_plan
            != execution_plan(&prepared.request, expected.identity.digest())?
    {
        return Err(invalid(
            "random-response recovery import resource authority differs",
        ));
    }
    crate::field_recovery::import_recovered_fields(
        source,
        response,
        &prepared.request,
        &prepared.host,
        &prepared.program,
        limits,
    )
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
