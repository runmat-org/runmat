//! Source-bound retained maxwell field tasks; never re-solve or reconstruct lost fields.
use crate::maxwell_source_admission::authenticate_maxwell_source_objects as authenticate_objects;
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisTransportProfiler, AnalysisWorkKind,
        ExecutionPlanIdentity, FieldRecoveryRequest, MaxwellHarmonicSolverResult,
        MaxwellHarmonicWorkload, NumericalPlanIdentity,
    },
    planning::resource::{
        build_maxwell_field_recovery_resource_plan, maxwell_field_recovery_partition_count,
        maxwell_field_recovery_shape, maxwell_recovery_source_validation_workspace, ByteCount,
        FeaResourcePlan, PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct MaxwellFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}
pub struct PreparedMaxwellFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub fn prepare_maxwell_field_recovery(
    p: MaxwellFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedMaxwellFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveMaxwellHarmonic
        || p.source_host.workload.inputs.len() != 3
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
            "maxwell recovery requires its completed maxwell source",
        ));
    }
    for (kind, objects) in [
        (
            AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
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
            return Err(invalid("maxwell recovery original source closure differs"));
        }
    }
    authenticate_objects(
        &p.workload,
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
    )?;
    authenticate_objects(&p.mesh, AnalysisNumericalObjectKind::Mesh)?;
    authenticate_objects(&p.source_result, AnalysisNumericalObjectKind::SolverResult)?;
    let workload = MaxwellHarmonicWorkload::from_canonical_bytes(&p.workload.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&p.mesh.payload()).map_err(invalid)?;
    let result = decode_maxwell_recovery_source(&p.workload, &p.mesh, &p.source_result)?;
    let original_plan = p
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .ok_or_else(|| invalid("source maxwell task omits resource plan"))?;
    let numerical = NumericalPlanIdentity::from_maxwell_harmonic_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if p.source_host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, original_plan.logical_identity)?
        || result.producer_identity() != p.source_host.workload.step
    {
        return Err(invalid("source maxwell execution lineage differs"));
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
    let resource_plan = build_prepared_maxwell_field_recovery_resource_plan(
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
        inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if input.manifest.kind == AnalysisNumericalObjectKind::SolverResult
                {
                    ValueRefKind::ResultObject
                } else {
                    ValueRefKind::DriverObject
                },
            })
            .collect(),
        vec![AnalysisNumericalObjectKind::Field],
        p.source_host.workload.required_capabilities.clone(),
    )?;
    let host =
        AnalysisHostWorkload::new(request_host, p.source_host.artifact_access, p.output_policy)?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedMaxwellFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
        local_execution: p.local_execution,
    })
}

/// Authenticate original transport before workload/mesh decoding; no result decode.
pub fn build_prepared_maxwell_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let kinds = [
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
        AnalysisNumericalObjectKind::Mesh,
        AnalysisNumericalObjectKind::SolverResult,
    ];
    if inputs.len() < 4
        || inputs.len() > 5
        || kinds
            .iter()
            .any(|kind| inputs.iter().filter(|i| i.manifest.kind == *kind).count() != 1)
        || inputs
            .iter()
            .filter(|i| i.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .count()
            != inputs.len() - 4
    {
        return Err(invalid(
            "Maxwell recovery immutable input inventory differs before source decode",
        ));
    }
    for input in inputs {
        authenticate_objects(input, input.manifest.kind)?;
    }
    let unique = |kind| {
        inputs
            .iter()
            .find(|i| i.manifest.kind == kind)
            .ok_or_else(|| invalid("maxwell recovery source absent"))
    };
    let workload = MaxwellHarmonicWorkload::from_canonical_bytes(
        &unique(AnalysisNumericalObjectKind::MaxwellHarmonicWorkload)?.payload(),
    )?;
    let mesh =
        SolverMeshArtifact::canonical_decode(&unique(AnalysisNumericalObjectKind::Mesh)?.payload())
            .map_err(invalid)?;
    if workload.identity()?.digest()
        != unique(AnalysisNumericalObjectKind::MaxwellHarmonicWorkload)?
            .manifest
            .logical_identity
        || Digest::from_bytes(*mesh.canonical_digest.bytes())
            != unique(AnalysisNumericalObjectKind::Mesh)?
                .manifest
                .logical_identity
    {
        return Err(invalid(
            "Maxwell recovery original workload or mesh identity differs before projection",
        ));
    }
    let workspace = maxwell_recovery_source_validation_workspace(&workload, &mesh)?;
    build_maxwell_recovery_metadata_resource_plan(
        request,
        inputs,
        policy,
        workspace,
        local_execution,
    )
}

/// Authenticate immutable closure profiles and output policy before domain decoding.
/// The carried source bound is then rederived from admitted workload/mesh before
/// allocating any derived property/projection/support maps or decoding the result.
pub fn build_maxwell_recovery_metadata_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    source_validation_workspace: ByteCount,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let numerical = inputs
        .iter()
        .filter(|i| i.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    let kinds = [
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
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
            "maxwell recovery immutable input inventory differs",
        ));
    }
    for input in &numerical {
        authenticate_objects(input, input.manifest.kind)?;
    }
    let unique = |kind| {
        numerical
            .iter()
            .copied()
            .find(|i| i.manifest.kind == kind)
            .ok_or_else(|| invalid("missing maxwell recovery input"))
    };
    if unique(AnalysisNumericalObjectKind::FieldRecoveryRequest)?
        .manifest
        .logical_identity
        != request.identity().digest()
        || Some(
            unique(AnalysisNumericalObjectKind::SolverResult)?
                .manifest
                .logical_identity,
        ) != request.source.solver_result().map(|i| i.digest())
    {
        return Err(invalid("maxwell recovery request source differs"));
    }
    if FieldRecoveryRequest::from_canonical_bytes(
        &unique(AnalysisNumericalObjectKind::FieldRecoveryRequest)?.payload(),
    )? != *request
    {
        return Err(invalid(
            "Maxwell recovery encoded request differs from admitted request",
        ));
    }
    // Every nonempty partition requires at least one transport object, plus
    // the root manifest. Reject impossible inventories before profiling values.
    if maxwell_field_recovery_partition_count(request)? >= policy.inventory_limits().max_objects {
        return Err(invalid(
            "maxwell recovery partition inventory exceeds output policy",
        ));
    }
    let (partitions, _) = maxwell_field_recovery_shape(request)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    Ok(build_maxwell_field_recovery_resource_plan(
        request,
        source_validation_workspace,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_transport_output_resource_facts(&profile)?,
            output_set: projected_output_set_resource_facts(),
        },
        local_execution,
    )?)
}

pub fn decode_maxwell_recovery_source(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<MaxwellHarmonicSolverResult> {
    authenticate_objects(
        workload,
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
    )?;
    authenticate_objects(mesh, AnalysisNumericalObjectKind::Mesh)?;
    authenticate_objects(source, AnalysisNumericalObjectKind::SolverResult)?;
    let w = MaxwellHarmonicWorkload::from_canonical_bytes(&workload.payload())?;
    let m = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    if workload.manifest.logical_identity != w.identity()?.digest()
        || mesh.manifest.logical_identity != Digest::from_bytes(*m.canonical_digest.bytes())
    {
        return Err(invalid(
            "retained Maxwell original workload or mesh identity differs",
        ));
    }
    maxwell_recovery_source_validation_workspace(&w, &m)?;
    let r = MaxwellHarmonicSolverResult::from_canonical_bytes_for(
        &source.payload(),
        source.manifest.payload_digest,
        &w,
        &m,
    )?;
    if source.manifest.logical_identity != r.identity()?.digest() {
        return Err(invalid("retained maxwell source identity differs"));
    }
    Ok(r)
}
fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}

pub fn import_maxwell_recovered_fields(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    prepared: &PreparedMaxwellFieldRecovery,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    response.validate_against(&prepared.program.request)?;
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
    {
        return Err(invalid(
            "maxwell field import requires its exact recovery task",
        ));
    }
    let source_input = |kind| {
        prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .ok_or_else(|| invalid("maxwell recovery import source absent"))
    };
    let host_inputs = prepared
        .inputs
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
        .collect::<Vec<_>>();
    if host_inputs != prepared.host.workload.inputs {
        return Err(invalid(
            "maxwell field import closure differs from admitted task",
        ));
    }
    let expected = build_prepared_maxwell_field_recovery_resource_plan(
        &prepared.request,
        &prepared.inputs,
        prepared.output_policy,
        prepared.local_execution,
    )?;
    if expected != prepared.resource_plan {
        return Err(invalid(
            "maxwell recovery import resource authority differs",
        ));
    }
    let encoded_plan = source_input(AnalysisNumericalObjectKind::ResourcePlan)?;
    if FeaResourcePlan::from_canonical_bytes(&encoded_plan.payload())? != expected
        || encoded_plan.manifest.logical_identity != expected.identity.digest()
    {
        return Err(invalid(
            "Maxwell recovery original plan differs before retained source validation",
        ));
    }
    let result = decode_maxwell_recovery_source(
        source_input(AnalysisNumericalObjectKind::MaxwellHarmonicWorkload)?,
        source_input(AnalysisNumericalObjectKind::Mesh)?,
        source_input(AnalysisNumericalObjectKind::SolverResult)?,
    )?;
    if prepared.request != result.native_field_request_from_result()? {
        return Err(invalid(
            "Maxwell field import differs from original native inventory",
        ));
    }
    let numerical = NumericalPlanIdentity::from_maxwell_harmonic_workload(
        result.source_workload(),
        Digest::from_bytes(*result.response().mesh_digest.bytes()),
    )?;
    if prepared.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, expected.identity.digest())?
    {
        return Err(invalid(
            "maxwell field import numerical execution authority differs",
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
