//! Source-bound retained electrical field task; never re-solves or reconstructs lost fields.
//!
//! A static electrical solve publishes its nodal potential and element field, current density
//! (or displacement) and loss (or energy) density as the shared recovered-field contract, so
//! authored observations evaluate them.
use crate::portable_source_admission::authenticate_source_objects as authenticate_objects;
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisTransportProfiler, AnalysisWorkKind,
        ElectricalStaticSolverResult, ElectricalStaticWorkload, ExecutionPlanIdentity,
        FieldRecoveryRequest, NumericalPlanIdentity,
    },
    planning::resource::{
        build_electrical_field_recovery_resource_plan, electrical_field_recovery_partition_count,
        electrical_field_recovery_shape, electrical_recovery_source_validation_workspace,
        ByteCount, FeaResourcePlan, PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct ElectricalFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub struct PreparedElectricalFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    /// The retained source, decoded and validated once at preparation; import reuses it.
    pub source: ElectricalRecoverySource,
}

/// Retained electrical static source (workload, mesh, result), decoded from its transported
/// objects, authenticated and validated against each other exactly once. Sealed: only
/// [`decode_electrical_recovery_source`] builds it.
pub struct ElectricalRecoverySource {
    workload: ElectricalStaticWorkload,
    mesh: SolverMeshArtifact,
    result: ElectricalStaticSolverResult,
}

impl ElectricalRecoverySource {
    pub fn workload(&self) -> &ElectricalStaticWorkload {
        &self.workload
    }

    pub fn mesh(&self) -> &SolverMeshArtifact {
        &self.mesh
    }

    pub fn result(&self) -> &ElectricalStaticSolverResult {
        &self.result
    }

    /// The native recovery request of a result already validated against its workload and mesh.
    pub fn native_field_request(&self) -> AnalysisBridgeResult<FieldRecoveryRequest> {
        Ok(self.result.native_field_request_from_result()?)
    }
}

const SOURCE_KINDS: [AnalysisNumericalObjectKind; 4] = [
    AnalysisNumericalObjectKind::FieldRecoveryRequest,
    AnalysisNumericalObjectKind::ElectricalStaticWorkload,
    AnalysisNumericalObjectKind::Mesh,
    AnalysisNumericalObjectKind::SolverResult,
];

pub fn prepare_electrical_field_recovery(
    p: ElectricalFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedElectricalFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveElectricalStatic
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
            "electrical recovery requires its completed electrical static source",
        ));
    }
    for (kind, objects) in [
        (
            AnalysisNumericalObjectKind::ElectricalStaticWorkload,
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
                "electrical recovery original source closure differs",
            ));
        }
    }
    let source = decode_electrical_recovery_source(&p.workload, &p.mesh, &p.source_result)?;
    let (workload, mesh, result) = (source.workload(), source.mesh(), source.result());
    let original_plan = p
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .ok_or_else(|| invalid("source electrical task omits resource plan"))?;
    let numerical = NumericalPlanIdentity::from_electrical_static_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if p.source_host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, original_plan.logical_identity)?
        || result.producer() != p.source_host.workload.step
    {
        return Err(invalid("source electrical execution lineage differs"));
    }
    let request = source.native_field_request()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        request.identity().digest(),
        &request.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?;
    let mut inputs = vec![request_objects, p.workload, p.mesh, p.source_result];
    let resource_plan = electrical_field_recovery_resource_plan_on(
        &request,
        &inputs,
        source.workload(),
        source.mesh(),
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
    Ok(PreparedElectricalFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
        local_execution: p.local_execution,
        source,
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
pub fn build_prepared_electrical_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let unique = |kind| {
        inputs
            .iter()
            .find(|i| i.manifest.kind == kind)
            .ok_or_else(|| invalid("missing electrical recovery input"))
    };
    // Workload and mesh only: the result is decoded after the plan is admitted.
    let workload = ElectricalStaticWorkload::from_canonical_bytes(
        &unique(AnalysisNumericalObjectKind::ElectricalStaticWorkload)?.payload(),
    )?;
    let mesh =
        SolverMeshArtifact::canonical_decode(&unique(AnalysisNumericalObjectKind::Mesh)?.payload())
            .map_err(invalid)?;
    electrical_field_recovery_resource_plan_on(
        request,
        inputs,
        &workload,
        &mesh,
        policy,
        local_execution,
    )
}

/// Resource plan of an electrical recovery whose source was decoded and validated once.
fn electrical_field_recovery_resource_plan_on(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
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
            "electrical recovery immutable input inventory differs",
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
            .ok_or_else(|| invalid("missing electrical recovery input"))
    };
    if workload.identity()?.digest()
        != unique(AnalysisNumericalObjectKind::ElectricalStaticWorkload)?
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
            "electrical recovery request, workload or mesh differs from its admitted closure",
        ));
    }
    if electrical_field_recovery_partition_count(request)? >= policy.inventory_limits().max_objects
    {
        return Err(invalid(
            "electrical recovery partition inventory exceeds output policy",
        ));
    }
    let (partitions, _) = electrical_field_recovery_shape(request)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    let workspace: ByteCount = electrical_recovery_source_validation_workspace(workload, mesh)?;
    Ok(build_electrical_field_recovery_resource_plan(
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

/// Decode and authenticate the retained electrical static result against its workload and mesh.
pub fn decode_electrical_recovery_source(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<ElectricalRecoverySource> {
    authenticate_objects(
        workload,
        AnalysisNumericalObjectKind::ElectricalStaticWorkload,
    )?;
    authenticate_objects(mesh, AnalysisNumericalObjectKind::Mesh)?;
    authenticate_objects(source, AnalysisNumericalObjectKind::SolverResult)?;
    let w = ElectricalStaticWorkload::from_canonical_bytes(&workload.payload())?;
    let m = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    m.validate_canonical().map_err(invalid)?;
    if workload.manifest.logical_identity != w.identity()?.digest()
        || mesh.manifest.logical_identity != Digest::from_bytes(*m.canonical_digest.bytes())
    {
        return Err(invalid(
            "retained electrical original workload or mesh identity differs",
        ));
    }
    let r = ElectricalStaticSolverResult::from_canonical_bytes(&source.payload())?;
    r.validate_for(&w, &m)?;
    if source.manifest.logical_identity != r.source_result_identity()?.digest() {
        return Err(invalid("retained electrical source identity differs"));
    }
    Ok(ElectricalRecoverySource {
        workload: w,
        mesh: m,
        result: r,
    })
}

pub fn import_electrical_recovered_fields(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    prepared: &PreparedElectricalFieldRecovery,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    response.validate_against(&prepared.program.request)?;
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
        || host_inputs(&prepared.inputs) != prepared.host.workload.inputs
    {
        return Err(invalid(
            "electrical field import requires its exact recovery task",
        ));
    }
    let expected = electrical_field_recovery_resource_plan_on(
        &prepared.request,
        &prepared.inputs,
        prepared.source.workload(),
        prepared.source.mesh(),
        prepared.output_policy,
        prepared.local_execution,
    )?;
    if expected != prepared.resource_plan {
        return Err(invalid(
            "electrical recovery import resource authority differs",
        ));
    }
    let input = |kind| {
        prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .ok_or_else(|| invalid("electrical recovery import source absent"))
    };
    let encoded_plan = input(AnalysisNumericalObjectKind::ResourcePlan)?;
    if FeaResourcePlan::from_canonical_bytes(&encoded_plan.payload())? != expected
        || encoded_plan.manifest.logical_identity != expected.identity.digest()
    {
        return Err(invalid(
            "electrical recovery original plan differs before retained source validation",
        ));
    }
    // The inputs are the exact objects the source was decoded and validated from at preparation
    // (`electrical_field_recovery_resource_plan_on` re-authenticated them above).
    for (kind, digest) in [
        (
            AnalysisNumericalObjectKind::SolverResult,
            prepared.source.result().source_result_identity()?.digest(),
        ),
        (
            AnalysisNumericalObjectKind::Mesh,
            Digest::from_bytes(*prepared.source.mesh().canonical_digest.bytes()),
        ),
    ] {
        if input(kind)?.manifest.logical_identity != digest {
            return Err(invalid(
                "electrical field import source differs from its preparation",
            ));
        }
    }
    let (workload, mesh) = (prepared.source.workload(), prepared.source.mesh());
    if prepared.request != prepared.source.native_field_request()? {
        return Err(invalid(
            "electrical field import differs from original native inventory",
        ));
    }
    let numerical = NumericalPlanIdentity::from_electrical_static_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if prepared.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, expected.identity.digest())?
    {
        return Err(invalid(
            "electrical field import numerical execution authority differs",
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
