//! Source-bound native pressure copies for a canonical acoustic modal or transient result.
//! Authored observations consume this shared field artifact; no field is reconstructed by a
//! driver-side solve. The original result, workload and mesh are reauthenticated at every edge.
use crate::portable_source_admission::authenticate_source_objects;
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AcousticResponseSolverResult, AcousticResponseWorkload,
        AnalysisNumericalObjectKind as Kind, AnalysisTransportProfiler, AnalysisWorkKind,
        ExecutionPlanIdentity, FieldRecoveryRequest, NumericalPlanIdentity,
    },
    planning::resource::{
        acoustic_response_field_recovery_shape,
        acoustic_response_recovery_source_validation_workspace,
        acoustic_response_result_payload_bytes_upper_bound,
        build_acoustic_response_field_recovery_resource_plan, FeaResourcePlan,
        PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ObjectNamespace, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct AcousticResponseFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub struct PreparedAcousticResponseFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
fn input(
    inputs: &[PreparedAnalysisObjects],
    kind: Kind,
) -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
    let mut matches = inputs.iter().filter(|input| input.manifest.kind == kind);
    match (matches.next(), matches.next()) {
        (Some(value), None) => Ok(value),
        _ => Err(invalid(
            "acoustic response recovery requires one input of each declared kind",
        )),
    }
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
            reference_kind: if input.manifest.kind == Kind::SolverResult {
                ValueRefKind::ResultObject
            } else {
                ValueRefKind::DriverObject
            },
        })
        .collect()
}

pub fn decode_acoustic_response_recovery_source(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<(
    AcousticResponseWorkload,
    SolverMeshArtifact,
    AcousticResponseSolverResult,
)> {
    authenticate_source_objects(workload, Kind::AcousticResponseWorkload)?;
    authenticate_source_objects(mesh, Kind::Mesh)?;
    authenticate_source_objects(source, Kind::SolverResult)?;
    let w = AcousticResponseWorkload::from_canonical_bytes(&workload.payload())?;
    let m = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    m.validate_canonical().map_err(invalid)?;
    if workload.manifest.logical_identity != w.identity()?.digest()
        || mesh.manifest.logical_identity != Digest::from_bytes(*m.canonical_digest.bytes())
    {
        return Err(invalid(
            "acoustic response recovery workload or mesh identity differs",
        ));
    }
    let maximum = acoustic_response_result_payload_bytes_upper_bound(&w, &m)?;
    if source.manifest.encoded_length > maximum {
        return Err(invalid(
            "acoustic response recovery source exceeds admitted result bound",
        ));
    }
    let r = AcousticResponseSolverResult::from_canonical_bytes_for(&source.payload(), &w, &m)?;
    if source.manifest.logical_identity != r.source_result_identity()?.digest() {
        return Err(invalid(
            "acoustic response recovery source result identity differs",
        ));
    }
    Ok((w, m, r))
}

pub fn build_prepared_acoustic_response_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let numerical_count = inputs
        .iter()
        .filter(|input| input.manifest.kind != Kind::ResourcePlan)
        .count();
    if numerical_count != 4 || inputs.len() - numerical_count > 1 {
        return Err(invalid(
            "acoustic response recovery input inventory differs",
        ));
    }
    for objects in inputs {
        authenticate_source_objects(objects, objects.manifest.kind)?;
    }
    let request_input = input(inputs, Kind::FieldRecoveryRequest)?;
    if FieldRecoveryRequest::from_canonical_bytes(&request_input.payload())? != *request
        || request_input.manifest.logical_identity != request.identity().digest()
    {
        return Err(invalid(
            "acoustic response recovery request differs from immutable input",
        ));
    }
    let (workload, mesh, result) = decode_acoustic_response_recovery_source(
        input(inputs, Kind::AcousticResponseWorkload)?,
        input(inputs, Kind::Mesh)?,
        input(inputs, Kind::SolverResult)?,
    )?;
    if request != &result.native_field_request_from_result()? {
        return Err(invalid(
            "acoustic response recovery request differs from retained pressure",
        ));
    }
    let (partitions, _) = acoustic_response_field_recovery_shape(request, &result)?;
    if partitions.len() >= policy.inventory_limits().max_objects {
        return Err(invalid(
            "acoustic response recovery partition inventory exceeds policy",
        ));
    }
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(Kind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    let source_validation_workspace_bytes =
        acoustic_response_recovery_source_validation_workspace(&workload, &mesh)?;
    Ok(build_acoustic_response_field_recovery_resource_plan(
        request,
        &result,
        source_validation_workspace_bytes,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_transport_output_resource_facts(&profile)?,
            output_set: projected_output_set_resource_facts(),
        },
        local_execution,
    )?)
}

pub fn prepare_acoustic_response_field_recovery(
    p: AcousticResponseFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedAcousticResponseFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveAcousticResponse
        || p.source_host.workload.outputs
            != [
                Kind::SolverResult,
                Kind::NumericalReceipt,
                Kind::StatePublicationProfiles,
            ]
        || p.source_host.workload.inputs.len() != 3
    {
        return Err(invalid(
            "acoustic response recovery requires completed original solve",
        ));
    }
    for objects in [&p.workload, &p.mesh] {
        if p.source_host
            .workload
            .inputs
            .iter()
            .filter(|input| {
                input.kind == objects.manifest.kind
                    && input.logical_identity == objects.manifest.logical_identity
                    && input.payload_digest == objects.manifest.payload_digest
                    && input.manifest_digest == objects.root.digest
            })
            .count()
            != 1
        {
            return Err(invalid(
                "acoustic response recovery original input closure differs",
            ));
        }
    }
    let (workload, mesh, result) =
        decode_acoustic_response_recovery_source(&p.workload, &p.mesh, &p.source_result)?;
    let original_plan = p
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == Kind::ResourcePlan)
        .ok_or_else(|| invalid("acoustic response original resource plan absent"))?;
    let numerical = NumericalPlanIdentity::from_acoustic_response_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if p.source_host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, original_plan.logical_identity)?
        || p.source_host.workload.step != result.producer()
    {
        return Err(invalid(
            "acoustic response recovery original numerical lineage differs",
        ));
    }
    let request = result.native_field_request_from_result()?;
    let request_objects = prepare_analysis_objects(
        Kind::FieldRecoveryRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?;
    let mut inputs = vec![request_objects, p.workload, p.mesh, p.source_result];
    let resource_plan = build_prepared_acoustic_response_field_recovery_resource_plan(
        &request,
        &inputs,
        p.output_policy,
        p.local_execution,
    )?;
    inputs.push(prepare_analysis_objects(
        Kind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?);
    let host = AnalysisHostWorkload::new(
        AnalysisWorkloadRequest::new(
            p.source_host.workload.study,
            p.source_host.workload.step,
            ExecutionPlanIdentity::new(numerical, resource_plan.identity.digest())?,
            AnalysisWorkKind::RecoverField,
            host_inputs(&inputs),
            vec![Kind::Field],
            p.source_host.workload.required_capabilities.clone(),
        )?,
        p.source_host.artifact_access,
        p.output_policy,
    )?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedAcousticResponseFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
        local_execution: p.local_execution,
    })
}

pub fn import_acoustic_response_recovered_fields(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedAcousticResponseFieldRecovery,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    response.validate_against(&prepared.program.request)?;
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [Kind::Field]
        || host_inputs(&prepared.inputs) != prepared.host.workload.inputs
    {
        return Err(invalid("acoustic response recovery import task differs"));
    }
    let expected = build_prepared_acoustic_response_field_recovery_resource_plan(
        &prepared.request,
        &prepared.inputs,
        prepared.output_policy,
        prepared.local_execution,
    )?;
    if expected != prepared.resource_plan
        || FeaResourcePlan::from_canonical_bytes(
            &input(&prepared.inputs, Kind::ResourcePlan)?.payload(),
        )? != expected
    {
        return Err(invalid(
            "acoustic response recovery import resource authority differs",
        ));
    }
    let (workload, mesh, result) = decode_acoustic_response_recovery_source(
        input(&prepared.inputs, Kind::AcousticResponseWorkload)?,
        input(&prepared.inputs, Kind::Mesh)?,
        input(&prepared.inputs, Kind::SolverResult)?,
    )?;
    let numerical = NumericalPlanIdentity::from_acoustic_response_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if prepared.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, expected.identity.digest())?
        || prepared.request != result.native_field_request_from_result()?
    {
        return Err(invalid("acoustic response recovery import source differs"));
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
