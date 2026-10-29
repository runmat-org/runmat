//! Admitted original J2 field publication; no private numerical or observation authority.
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, AnalysisTransportProfiler, AnalysisWorkKind,
        ExecutionPlanIdentity, FieldRecoveryRequest, NonlinearSolidBlockArtifact,
        NonlinearSolidWorkload, NumericalPlanIdentity,
    },
    planning::resource::{
        build_nonlinear_field_recovery_resource_plan, nonlinear_field_recovery_shape,
        nonlinear_recovery_workspace, ByteCount, FeaResourcePlan, PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
mod admission;
pub use admission::require_nonlinear_recovery_decode_budget;

pub struct NonlinearFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_block: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}
pub struct PreparedNonlinearFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
fn unique(
    inputs: &[PreparedAnalysisObjects],
    kind: Kind,
) -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
    let mut items = inputs.iter().filter(|i| i.manifest.kind == kind);
    let item = items
        .next()
        .ok_or_else(|| invalid("original nonlinear recovery input absent"))?;
    if items.next().is_some() {
        return Err(invalid("nonlinear recovery input duplicated"));
    }
    Ok(item)
}

pub fn decode_nonlinear_recovery_source(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<(
    NonlinearSolidWorkload,
    SolverMeshArtifact,
    NonlinearSolidBlockArtifact,
)> {
    let w = unique(inputs, Kind::NonlinearSolidWorkload)?;
    let m = unique(inputs, Kind::Mesh)?;
    let b = unique(inputs, Kind::NonlinearSolidBlockArtifact)?;
    for input in [w, m, b] {
        crate::portable_source_admission::authenticate_source_objects(input, input.manifest.kind)?;
    }
    let workload = NonlinearSolidWorkload::from_canonical_bytes(&w.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&m.payload()).map_err(invalid)?;
    let block = NonlinearSolidBlockArtifact::from_canonical_bytes(
        &b.payload(),
        b.manifest.logical_identity,
    )?;
    if workload.identity()?.digest() != w.manifest.logical_identity
        || m.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
    {
        return Err(invalid(
            "original nonlinear recovery workload or mesh identity differs",
        ));
    }
    block.native_field_request(&workload, &mesh)?;
    Ok((workload, mesh, block))
}

pub fn prepare_nonlinear_field_recovery(
    p: NonlinearFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedNonlinearFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveNonlinearSolidBlock {
        return Err(invalid(
            "nonlinear recovery requires original completed block task",
        ));
    }
    for objects in [&p.workload, &p.mesh] {
        let matches = p
            .source_host
            .workload
            .inputs
            .iter()
            .filter(|i| i.kind == objects.manifest.kind)
            .collect::<Vec<_>>();
        if matches.len() != 1
            || matches[0].logical_identity != objects.manifest.logical_identity
            || matches[0].payload_digest != objects.manifest.payload_digest
            || matches[0].manifest_digest != objects.root.digest
            || matches[0].manifest_encoded_length != objects.root.encoded_length
        {
            return Err(invalid("nonlinear recovery original input closure differs"));
        }
    }
    let mut inputs = vec![p.workload, p.mesh, p.source_block];
    let (workload, mesh, block) = decode_nonlinear_recovery_source(&inputs)?;
    let original_plan = p
        .source_host
        .workload
        .inputs
        .iter()
        .find(|i| i.kind == Kind::ResourcePlan)
        .ok_or_else(|| invalid("original nonlinear source omits resource plan"))?;
    let numerical = NumericalPlanIdentity::from_nonlinear_solid_block(&block.request)?;
    if p.source_host.workload.step != workload.producer_identity()?
        || p.source_host.workload.execution_plan
            != ExecutionPlanIdentity::new(numerical, original_plan.logical_identity)?
    {
        return Err(invalid(
            "nonlinear recovery original task authority differs",
        ));
    }
    let request = block.native_field_request(&workload, &mesh)?;
    inputs.push(prepare_analysis_objects(
        Kind::FieldRecoveryRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?);
    let resource_plan = build_prepared_nonlinear_recovery_resource_plan(
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
    let workload = AnalysisWorkloadRequest::new(
        p.source_host.workload.study,
        p.source_host.workload.step,
        ExecutionPlanIdentity::new(numerical, resource_plan.identity.digest())?,
        AnalysisWorkKind::RecoverField,
        inputs.iter().map(input_reference).collect(),
        vec![Kind::Field],
        p.source_host.workload.required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(workload, p.source_host.artifact_access, p.output_policy)?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedNonlinearFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
        local_execution: p.local_execution,
    })
}

fn input_reference(i: &PreparedAnalysisObjects) -> AnalysisWorkloadInput {
    AnalysisWorkloadInput {
        kind: i.manifest.kind,
        logical_identity: i.manifest.logical_identity,
        payload_digest: i.manifest.payload_digest,
        manifest_digest: i.root.digest,
        manifest_encoded_length: i.root.encoded_length,
        reference_kind: if i.manifest.kind == Kind::NonlinearSolidBlockArtifact {
            ValueRefKind::ResultObject
        } else {
            ValueRefKind::DriverObject
        },
    }
}

pub fn build_prepared_nonlinear_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let (workload, mesh, block) = decode_nonlinear_recovery_source(inputs)?;
    if *request != block.native_field_request(&workload, &mesh)? {
        return Err(invalid(
            "nonlinear recovery request differs before physical projection",
        ));
    }
    build_nonlinear_recovery_metadata_resource_plan(
        request,
        inputs,
        policy,
        nonlinear_recovery_workspace(&workload, &mesh, &block)?,
        local_execution,
    )
}

pub fn build_nonlinear_recovery_metadata_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    workspace: ByteCount,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let kinds = [
        Kind::FieldRecoveryRequest,
        Kind::NonlinearSolidWorkload,
        Kind::Mesh,
        Kind::NonlinearSolidBlockArtifact,
    ];
    let numerical = inputs
        .iter()
        .filter(|i| i.manifest.kind != Kind::ResourcePlan)
        .collect::<Vec<_>>();
    if numerical.len() != kinds.len()
        || inputs.len() - numerical.len() > 1
        || kinds.iter().any(|kind| {
            numerical
                .iter()
                .filter(|i| i.manifest.kind == *kind)
                .count()
                != 1
        })
        || unique(inputs, Kind::FieldRecoveryRequest)?
            .manifest
            .logical_identity
            != request.identity().digest()
    {
        return Err(invalid(
            "nonlinear recovery immutable input kinds or request differ",
        ));
    }
    let (partitions, _) = nonlinear_field_recovery_shape(request)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(Kind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    Ok(build_nonlinear_field_recovery_resource_plan(
        request,
        workspace,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_transport_output_resource_facts(&profile)?,
            output_set: projected_output_set_resource_facts(),
        },
        local_execution,
    )?)
}

pub fn import_nonlinear_recovered_fields(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedNonlinearFieldRecovery,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    require_nonlinear_recovery_decode_budget(
        &prepared.inputs,
        &prepared.resource_plan,
        prepared.output_policy.inventory_limits(),
    )?;
    let (workload, _mesh, block) = decode_nonlinear_recovery_source(&prepared.inputs)?;
    if prepared.resource_plan
        != build_prepared_nonlinear_recovery_resource_plan(
            &prepared.request,
            &prepared.inputs,
            prepared.output_policy,
            prepared.local_execution,
        )?
        || prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [Kind::Field]
        || prepared.host.workload.inputs
            != prepared
                .inputs
                .iter()
                .map(input_reference)
                .collect::<Vec<_>>()
        || prepared.host.workload.step != workload.producer_identity()?
        || prepared.host.workload.execution_plan
            != ExecutionPlanIdentity::new(
                NumericalPlanIdentity::from_nonlinear_solid_block(&block.request)?,
                prepared.resource_plan.identity.digest(),
            )?
    {
        return Err(invalid(
            "nonlinear recovery imported task authority differs",
        ));
    }
    let output = prepared
        .resource_plan
        .input
        .demand
        .output_facts(Kind::Field)
        .ok_or_else(|| invalid("nonlinear field output demand absent"))?;
    let mut limits = prepared.output_policy.inventory_limits();
    limits.max_total_bytes = limits
        .max_total_bytes
        .min(output.transport_closure_bytes_upper_bound.0);
    crate::field_recovery::import_recovered_fields(
        source,
        response,
        &prepared.request,
        &prepared.host,
        &prepared.program,
        limits,
    )
}
