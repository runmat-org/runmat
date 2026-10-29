//! Source-bound retained magnetic field tasks; never re-solve or reconstruct lost fields.
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisTransportProfiler, AnalysisWorkKind,
        ExecutionPlanIdentity, FieldRecoveryRequest, MagnetostaticSolverResult,
        MagnetostaticWorkload, NumericalPlanIdentity,
    },
    planning::resource::{
        build_magnetic_field_recovery_resource_plan, magnetic_field_recovery_partition_count,
        magnetic_field_recovery_shape, magnetic_recovery_source_validation_workspace, ByteCount,
        FeaResourcePlan, PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct MagneticFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}
pub struct PreparedMagneticFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub fn prepare_magnetic_field_recovery(
    p: MagneticFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedMagneticFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveMagnetostatic {
        return Err(invalid(
            "magnetic recovery requires its completed magnetic source",
        ));
    }
    for (kind, objects) in [
        (
            AnalysisNumericalObjectKind::MagnetostaticWorkload,
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
            return Err(invalid("magnetic recovery original source closure differs"));
        }
    }
    let workload = MagnetostaticWorkload::from_canonical_bytes(&p.workload.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&p.mesh.payload()).map_err(invalid)?;
    let result = decode_magnetic_recovery_source(&p.workload, &p.mesh, &p.source_result)?;
    let original_plan = p
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .ok_or_else(|| invalid("source magnetic task omits resource plan"))?;
    let numerical = NumericalPlanIdentity::from_magnetostatic_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if p.source_host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, original_plan.logical_identity)?
        || result.producer() != p.source_host.workload.step
    {
        return Err(invalid("source magnetic execution lineage differs"));
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
    let resource_plan = build_prepared_magnetic_field_recovery_resource_plan(
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
    Ok(PreparedMagneticFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
        local_execution: p.local_execution,
    })
}

/// Metadata-only rederivation runs before source workload, mesh and result decoding.
pub fn build_prepared_magnetic_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let unique = |kind| {
        inputs
            .iter()
            .find(|i| i.manifest.kind == kind)
            .ok_or_else(|| invalid("magnetic recovery source absent"))
    };
    let workload = MagnetostaticWorkload::from_canonical_bytes(
        &unique(AnalysisNumericalObjectKind::MagnetostaticWorkload)?.payload(),
    )?;
    let mesh =
        SolverMeshArtifact::canonical_decode(&unique(AnalysisNumericalObjectKind::Mesh)?.payload())
            .map_err(invalid)?;
    let workspace = magnetic_recovery_source_validation_workspace(&workload, &mesh)?;
    build_magnetic_recovery_metadata_resource_plan(
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
pub fn build_magnetic_recovery_metadata_resource_plan(
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
        AnalysisNumericalObjectKind::MagnetostaticWorkload,
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
            "magnetic recovery immutable input inventory differs",
        ));
    }
    let unique = |kind| {
        numerical
            .iter()
            .copied()
            .find(|i| i.manifest.kind == kind)
            .ok_or_else(|| invalid("missing magnetic recovery input"))
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
        return Err(invalid("magnetic recovery request source differs"));
    }
    // Every nonempty partition requires at least one transport object, plus
    // the root manifest. Reject impossible inventories before profiling values.
    if magnetic_field_recovery_partition_count(request)? >= policy.inventory_limits().max_objects {
        return Err(invalid(
            "magnetic recovery partition inventory exceeds output policy",
        ));
    }
    let (partitions, _) = magnetic_field_recovery_shape(request)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    Ok(build_magnetic_field_recovery_resource_plan(
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

pub fn decode_magnetic_recovery_source(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<MagnetostaticSolverResult> {
    let w = MagnetostaticWorkload::from_canonical_bytes(&workload.payload())?;
    let m = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    let r = MagnetostaticSolverResult::from_canonical_bytes(&source.payload())?;
    r.validate_for(&w, &m)?;
    if workload.manifest.logical_identity != w.identity()?.digest()
        || mesh.manifest.logical_identity != Digest::from_bytes(*m.canonical_digest.bytes())
        || source.manifest.kind != AnalysisNumericalObjectKind::SolverResult
        || source.manifest.logical_identity != r.source_result_identity()?.digest()
    {
        return Err(invalid("retained magnetic source identity differs"));
    }
    Ok(r)
}
fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}

pub fn import_magnetic_recovered_fields(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    prepared: &PreparedMagneticFieldRecovery,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
    {
        return Err(invalid(
            "magnetic field import requires its exact recovery task",
        ));
    }
    let source_input = |kind| {
        prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .ok_or_else(|| invalid("magnetic recovery import source absent"))
    };
    let result = decode_magnetic_recovery_source(
        source_input(AnalysisNumericalObjectKind::MagnetostaticWorkload)?,
        source_input(AnalysisNumericalObjectKind::Mesh)?,
        source_input(AnalysisNumericalObjectKind::SolverResult)?,
    )?;
    if prepared.request != result.native_field_request_from_result()? {
        return Err(invalid(
            "magnetic field import differs from original native inventory",
        ));
    }
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
            "magnetic field import closure differs from admitted task",
        ));
    }
    let expected = build_prepared_magnetic_field_recovery_resource_plan(
        &prepared.request,
        &prepared.inputs,
        prepared.output_policy,
        prepared.local_execution,
    )?;
    if expected != prepared.resource_plan {
        return Err(invalid(
            "magnetic recovery import resource authority differs",
        ));
    }
    let numerical = NumericalPlanIdentity::from_magnetostatic_workload(
        result.source_workload(),
        result.mesh_digest(),
    )?;
    let encoded_plan = source_input(AnalysisNumericalObjectKind::ResourcePlan)?;
    if FeaResourcePlan::from_canonical_bytes(&encoded_plan.payload())? != expected
        || encoded_plan.manifest.logical_identity != expected.identity.digest()
        || prepared.host.workload.execution_plan
            != ExecutionPlanIdentity::new(numerical, expected.identity.digest())?
    {
        return Err(invalid(
            "magnetic field import numerical execution authority differs",
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
