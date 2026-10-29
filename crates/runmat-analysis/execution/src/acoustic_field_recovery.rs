//! Source-bound retained acoustic field task; never re-solves or reconstructs lost fields.
//!
//! One entry of an authored frequency plan publishes its nodal pressure and element particle
//! velocity as the shared recovered-field contract, so authored observations evaluate them.
use crate::portable_source_admission::authenticate_source_objects as authenticate_objects;
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AcousticHarmonicSolverResult, AcousticHarmonicWorkload, AnalysisNumericalObjectKind,
        AnalysisTransportProfiler, AnalysisWorkKind, ExecutionPlanIdentity, FieldRecoveryRequest,
        NumericalPlanIdentity,
    },
    planning::resource::{
        acoustic_field_recovery_partition_count, acoustic_field_recovery_shape,
        acoustic_recovery_source_validation_workspace, build_acoustic_field_recovery_resource_plan,
        ByteCount, FeaResourcePlan, PreparedPhaseResourceFacts,
    },
    results::field::RecoveredFieldSample,
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct AcousticFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub struct PreparedAcousticFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

const SOURCE_KINDS: [AnalysisNumericalObjectKind; 4] = [
    AnalysisNumericalObjectKind::FieldRecoveryRequest,
    AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
    AnalysisNumericalObjectKind::Mesh,
    AnalysisNumericalObjectKind::SolverResult,
];

pub fn prepare_acoustic_field_recovery(
    p: AcousticFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedAcousticFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveAcousticHarmonic
        || p.source_host
            .workload
            .inputs
            .iter()
            .filter(|i| i.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .count()
            != 1
        || p.source_host.workload.outputs
            != [
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(invalid(
            "acoustic recovery requires its completed acoustic harmonic source",
        ));
    }
    for (kind, objects) in [
        (
            AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
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
            return Err(invalid("acoustic recovery original source closure differs"));
        }
    }
    let (workload, mesh, result) =
        decode_acoustic_recovery_source(&p.workload, &p.mesh, &p.source_result)?;
    let original_plan = p
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .ok_or_else(|| invalid("source acoustic task omits resource plan"))?;
    let numerical = NumericalPlanIdentity::from_acoustic_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if p.source_host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, original_plan.logical_identity)?
        || result.producer() != p.source_host.workload.step
    {
        return Err(invalid("source acoustic execution lineage differs"));
    }
    let request = result.native_field_request(&workload, &mesh)?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?;
    let mut inputs = vec![request_objects, p.workload, p.mesh, p.source_result];
    let resource_plan = build_prepared_acoustic_field_recovery_resource_plan(
        &request,
        &inputs,
        p.output_policy,
        p.local_execution,
    )?;
    inputs.push(prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?);
    let execution = ExecutionPlanIdentity::new(numerical, resource_plan.identity.digest())?;
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
    Ok(PreparedAcousticFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
        local_execution: p.local_execution,
    })
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

/// Authenticate the immutable closure and rederive the exact admitted plan. Workers call this
/// before allocating projected maps; the driver calls it before submitting the task.
pub fn build_prepared_acoustic_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let numerical = inputs
        .iter()
        .filter(|i| i.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    if numerical.len() != SOURCE_KINDS.len()
        || inputs.len() - numerical.len() > 1
        || SOURCE_KINDS.iter().any(|kind| {
            numerical
                .iter()
                .filter(|i| i.manifest.kind == *kind)
                .count()
                != 1
        })
    {
        return Err(invalid(
            "acoustic recovery immutable input inventory differs",
        ));
    }
    for input in inputs {
        authenticate_objects(input, input.manifest.kind)?;
    }
    let unique = |kind| {
        numerical
            .iter()
            .copied()
            .find(|i| i.manifest.kind == kind)
            .ok_or_else(|| invalid("missing acoustic recovery input"))
    };
    let workload = AcousticHarmonicWorkload::from_canonical_bytes(
        &unique(AnalysisNumericalObjectKind::AcousticHarmonicWorkload)?.payload(),
    )?;
    let mesh =
        SolverMeshArtifact::canonical_decode(&unique(AnalysisNumericalObjectKind::Mesh)?.payload())
            .map_err(invalid)?;
    if workload.identity()?.digest()
        != unique(AnalysisNumericalObjectKind::AcousticHarmonicWorkload)?
            .manifest
            .logical_identity
        || Digest::from_bytes(*mesh.canonical_digest.bytes())
            != unique(AnalysisNumericalObjectKind::Mesh)?
                .manifest
                .logical_identity
        || unique(AnalysisNumericalObjectKind::FieldRecoveryRequest)?
            .manifest
            .logical_identity
            != request.identity().digest()
        || Some(
            unique(AnalysisNumericalObjectKind::SolverResult)?
                .manifest
                .logical_identity,
        ) != request.source.solver_result().map(|i| i.digest())
        || FieldRecoveryRequest::from_canonical_bytes(
            &unique(AnalysisNumericalObjectKind::FieldRecoveryRequest)?.payload(),
        )? != *request
    {
        return Err(invalid(
            "acoustic recovery request, workload or mesh differs from its admitted closure",
        ));
    }
    if acoustic_field_recovery_partition_count(request)? >= policy.inventory_limits().max_objects {
        return Err(invalid(
            "acoustic recovery partition inventory exceeds output policy",
        ));
    }
    let sample = RecoveredFieldSample::FrequencyHertz(workload.frequency.canonical_value());
    let (partitions, _) = acoustic_field_recovery_shape(request, sample)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    let workspace: ByteCount = acoustic_recovery_source_validation_workspace(&workload, &mesh)?;
    Ok(build_acoustic_field_recovery_resource_plan(
        request,
        sample,
        workspace,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_transport_output_resource_facts(&profile)?,
            output_set: projected_output_set_resource_facts(),
        },
        local_execution,
    )?)
}

/// Decode and authenticate the retained acoustic harmonic result against its workload and mesh.
pub fn decode_acoustic_recovery_source(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<(
    AcousticHarmonicWorkload,
    SolverMeshArtifact,
    AcousticHarmonicSolverResult,
)> {
    authenticate_objects(
        workload,
        AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
    )?;
    authenticate_objects(mesh, AnalysisNumericalObjectKind::Mesh)?;
    authenticate_objects(source, AnalysisNumericalObjectKind::SolverResult)?;
    let w = AcousticHarmonicWorkload::from_canonical_bytes(&workload.payload())?;
    let m = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    m.validate_canonical().map_err(invalid)?;
    if workload.manifest.logical_identity != w.identity()?.digest()
        || mesh.manifest.logical_identity != Digest::from_bytes(*m.canonical_digest.bytes())
    {
        return Err(invalid(
            "retained acoustic original workload or mesh identity differs",
        ));
    }
    let r = AcousticHarmonicSolverResult::from_canonical_bytes(&source.payload())?;
    r.validate_for(&w, &m)?;
    if source.manifest.logical_identity != r.source_result_identity()?.digest() {
        return Err(invalid("retained acoustic source identity differs"));
    }
    Ok((w, m, r))
}

pub fn import_acoustic_recovered_fields(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    prepared: &PreparedAcousticFieldRecovery,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    response.validate_against(&prepared.program.request)?;
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
        || host_inputs(&prepared.inputs) != prepared.host.workload.inputs
    {
        return Err(invalid(
            "acoustic field import requires its exact recovery task",
        ));
    }
    let expected = build_prepared_acoustic_field_recovery_resource_plan(
        &prepared.request,
        &prepared.inputs,
        prepared.output_policy,
        prepared.local_execution,
    )?;
    if expected != prepared.resource_plan {
        return Err(invalid(
            "acoustic recovery import resource authority differs",
        ));
    }
    let input = |kind| {
        prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .ok_or_else(|| invalid("acoustic recovery import source absent"))
    };
    let encoded_plan = input(AnalysisNumericalObjectKind::ResourcePlan)?;
    if FeaResourcePlan::from_canonical_bytes(&encoded_plan.payload())? != expected
        || encoded_plan.manifest.logical_identity != expected.identity.digest()
    {
        return Err(invalid(
            "acoustic recovery original plan differs before retained source validation",
        ));
    }
    let (workload, mesh, result) = decode_acoustic_recovery_source(
        input(AnalysisNumericalObjectKind::AcousticHarmonicWorkload)?,
        input(AnalysisNumericalObjectKind::Mesh)?,
        input(AnalysisNumericalObjectKind::SolverResult)?,
    )?;
    if prepared.request != result.native_field_request(&workload, &mesh)? {
        return Err(invalid(
            "acoustic field import differs from original native inventory",
        ));
    }
    let numerical = NumericalPlanIdentity::from_acoustic_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if prepared.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, expected.identity.digest())?
    {
        return Err(invalid(
            "acoustic field import numerical execution authority differs",
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
