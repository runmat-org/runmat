//! Governed native copies of retained physical cell U/p and original owner-oriented face q.
//! Source validation evaluates original equations; this bridge never invokes the solver.
use crate::portable_source_admission::authenticate_source_objects;
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisTransportProfiler, AnalysisWorkKind,
        ExecutionPlanIdentity, FieldRecoveryRequest, FluidSteadySolverResult, FluidSteadyWorkload,
        NumericalPlanIdentity,
    },
    planning::resource::{
        build_fluid_field_recovery_resource_plan_on, fluid_field_recovery_shape, FeaResourcePlan,
        PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{
    object::{validate_inventory, ObjectInventoryLimits},
    ObjectNamespace,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

mod admission;

pub use admission::require_fluid_recovery_source_decode_budget;

pub struct FluidFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_resource_plan: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
}
pub struct PreparedFluidFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
}

fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
fn unique(
    inputs: &[PreparedAnalysisObjects],
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
    let mut matches = inputs.iter().filter(|i| i.manifest.kind == kind);
    let value = matches
        .next()
        .ok_or_else(|| invalid("fluid recovery source absent"))?;
    if matches.next().is_some() {
        return Err(invalid("duplicate fluid recovery source"));
    }
    Ok(value)
}
fn host_input(input: &PreparedAnalysisObjects) -> AnalysisWorkloadInput {
    AnalysisWorkloadInput {
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
    }
}

pub fn prepare_fluid_field_recovery(
    p: FluidFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedFluidFieldRecovery> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveFluidSteady
        || p.source_host.workload.outputs
            != [
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
            ]
        || p.source_host.workload.inputs.len() != 3
    {
        return Err(invalid(
            "fluid recovery requires its exact completed steady task",
        ));
    }
    for input in [&p.workload, &p.mesh, &p.source_resource_plan] {
        let expected = host_input(input);
        if p.source_host
            .workload
            .inputs
            .iter()
            .filter(|i| **i == expected)
            .count()
            != 1
        {
            return Err(invalid(
                "fluid recovery source differs from immutable solve input",
            ));
        }
        validate_inventory(&input.objects, p.output_policy.inventory_limits())?;
        authenticate_source_objects(input, input.manifest.kind)?;
    }
    validate_inventory(&p.source_result.objects, p.output_policy.inventory_limits())?;
    authenticate_source_objects(&p.source_result, AnalysisNumericalObjectKind::SolverResult)?;
    let source_plan = crate::decode_admitted_resource_plan(
        &p.source_resource_plan,
        p.output_policy.inventory_limits(),
    )?;
    let mut source_inputs = vec![p.workload, p.mesh, p.source_resource_plan];
    crate::require_fluid_source_decode_budget(
        &source_inputs,
        &source_plan,
        p.output_policy.inventory_limits(),
    )?;
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::FluidSteadySolve(source_demand) =
        &source_plan.input.demand
    else {
        return Err(invalid("fluid recovery original resource family differs"));
    };
    let source_closure = p
        .source_result
        .objects
        .iter()
        .try_fold(0_u64, |sum, object| {
            sum.checked_add(object.bytes.len() as u64)
                .ok_or_else(|| invalid("fluid source closure overflow"))
        })?;
    if p.source_result.manifest.encoded_length
        > source_demand
            .result_output
            .encoded_payload_bytes_upper_bound
            .0
        || source_closure
            > source_demand
                .result_output
                .transport_closure_bytes_upper_bound
                .0
    {
        return Err(invalid(
            "fluid original result exceeds solved output admission",
        ));
    }
    let source =
        decode_fluid_recovery_source_parts(&source_inputs[0], &source_inputs[1], &p.source_result)?;
    let (mesh, on_source) = &source;
    let (workload, result) = (on_source.workload(), on_source.result());
    let numerical = NumericalPlanIdentity::from_fluid_steady_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    let original_plan = p
        .source_host
        .workload
        .inputs
        .iter()
        .find(|i| i.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .ok_or_else(|| invalid("fluid source plan absent"))?;
    if p.source_host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, original_plan.logical_identity)?
        || p.source_host.workload.step != result.producer_identity()
    {
        return Err(invalid("fluid source numerical lineage differs"));
    }
    let request = on_source.native_field_request()?;
    let object = |kind, id, bytes: &[u8]| {
        prepare_analysis_objects(
            kind,
            id,
            bytes,
            p.output_policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            p.output_policy.inventory_limits(),
        )
    };
    let mut inputs = vec![
        object(
            AnalysisNumericalObjectKind::FieldRecoveryRequest,
            request.identity().digest(),
            &request.canonical_bytes()?,
        )?,
        source_inputs.remove(0),
        source_inputs.remove(0),
        p.source_result,
    ];
    let resource_plan = build_plan_with_source(&request, &inputs, p.output_policy, Some(&source))?;
    inputs.push(crate::portable_task_support::resource_plan_objects(
        &resource_plan,
        p.output_policy.chunk_bytes(),
        p.output_policy.inventory_limits(),
    )?);
    let host_request = AnalysisWorkloadRequest::new(
        p.source_host.workload.study,
        p.source_host.workload.step,
        ExecutionPlanIdentity::new(numerical, resource_plan.identity.digest())?,
        AnalysisWorkKind::RecoverField,
        inputs.iter().map(host_input).collect(),
        vec![AnalysisNumericalObjectKind::Field],
        p.source_host.workload.required_capabilities.clone(),
    )?;
    let host =
        AnalysisHostWorkload::new(host_request, p.source_host.artifact_access, p.output_policy)?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedFluidFieldRecovery {
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
    })
}

/// Memoized per exact request, input content and policy: every stage of a study asks for the
/// same plan, and each derivation decodes and re-prepares the original fluid source.
pub fn build_prepared_fluid_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    build_plan_with_source(request, inputs, policy, None)
}

/// The authenticated original mesh and the result validated against its workload's source.
type DecodedSource = (
    SolverMeshArtifact,
    runmat_analysis_fea::execution::FluidResultOnSource<'static>,
);

/// `decoded` is the source `decode_fluid_recovery_source_parts` already authenticated from
/// these same workload, mesh and result inputs; it is not decoded a second time.
fn build_plan_with_source(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    decoded: Option<&DecodedSource>,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    static MEMO: crate::fluid_phase_plan_memo::PlanMemo =
        crate::fluid_phase_plan_memo::PlanMemo::new();
    MEMO.get_or_derive(
        "fluid-field-recovery",
        &request.canonical_bytes()?,
        inputs,
        policy,
        || derive_prepared_fluid_field_recovery_resource_plan(request, inputs, policy, decoded),
    )
}

fn derive_prepared_fluid_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    decoded: Option<&DecodedSource>,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    admission::validate_inventory_shape(inputs)?;
    if inputs.len() == 5 {
        let carried = crate::decode_admitted_resource_plan(
            unique(inputs, AnalysisNumericalObjectKind::ResourcePlan)?,
            policy.inventory_limits(),
        )?;
        require_fluid_recovery_source_decode_budget(inputs, &carried, policy.inventory_limits())?;
    }
    for input in inputs {
        validate_inventory(&input.objects, policy.inventory_limits())?;
        authenticate_source_objects(input, input.manifest.kind)?;
    }
    let w_input = unique(inputs, AnalysisNumericalObjectKind::FluidSteadyWorkload)?;
    let m_input = unique(inputs, AnalysisNumericalObjectKind::Mesh)?;
    let r_input = unique(inputs, AnalysisNumericalObjectKind::SolverResult)?;
    let owned;
    let (mesh, on_source) = match decoded {
        Some(source) => source,
        None => {
            owned = decode_fluid_recovery_source_parts(w_input, m_input, r_input)?;
            &owned
        }
    };
    let (workload, result) = (on_source.workload(), on_source.result());
    if on_source.native_field_request()? != *request {
        return Err(invalid(
            "fluid recovery request differs from exact original native inventory",
        ));
    }
    for (kind, identity, bytes) in [
        (
            AnalysisNumericalObjectKind::FieldRecoveryRequest,
            request.identity().digest(),
            request.canonical_bytes()?,
        ),
        (
            AnalysisNumericalObjectKind::FluidSteadyWorkload,
            workload.identity()?.digest(),
            workload.canonical_bytes()?,
        ),
        (
            AnalysisNumericalObjectKind::Mesh,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            mesh.canonical_encode().map_err(invalid)?,
        ),
        (
            AnalysisNumericalObjectKind::SolverResult,
            result.identity()?.digest(),
            result.canonical_bytes()?,
        ),
    ] {
        let canonical = prepare_analysis_objects(
            kind,
            identity,
            &bytes,
            policy.chunk_bytes(),
            if kind == AnalysisNumericalObjectKind::SolverResult {
                ObjectNamespace::ResultValue
            } else {
                ObjectNamespace::InputValue
            },
            policy.inventory_limits(),
        )?;
        if unique(inputs, kind)? != &canonical {
            return Err(invalid(
                "fluid recovery source closure or chunk policy differs",
            ));
        }
    }
    let (partitions, _) = fluid_field_recovery_shape(request)?;
    if partitions.len() >= policy.inventory_limits().max_objects {
        return Err(invalid(
            "fluid native recovery exceeds output object inventory",
        ));
    }
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    let plan = build_fluid_field_recovery_resource_plan_on(
        on_source,
        mesh,
        request,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_transport_output_resource_facts(&profile)?,
            output_set: projected_output_set_resource_facts(),
        },
    )?;
    if inputs.len() == 5 {
        let canonical = crate::portable_task_support::resource_plan_objects(
            &plan,
            policy.chunk_bytes(),
            policy.inventory_limits(),
        )?;
        if unique(inputs, AnalysisNumericalObjectKind::ResourcePlan)? != &canonical {
            return Err(invalid(
                "fluid recovery immutable resource authority differs",
            ));
        }
    }
    Ok(plan)
}

pub fn decode_fluid_recovery_source(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<FluidSteadySolverResult> {
    Ok(decode_fluid_recovery_source_parts(workload, mesh, source)?
        .1
        .into_result())
}

/// The authenticated original workload, mesh and result, each decoded once.
fn decode_fluid_recovery_source_parts(
    workload: &PreparedAnalysisObjects,
    mesh: &PreparedAnalysisObjects,
    source: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<DecodedSource> {
    authenticate_source_objects(workload, AnalysisNumericalObjectKind::FluidSteadyWorkload)?;
    authenticate_source_objects(mesh, AnalysisNumericalObjectKind::Mesh)?;
    authenticate_source_objects(source, AnalysisNumericalObjectKind::SolverResult)?;
    let w = FluidSteadyWorkload::from_canonical_bytes(&workload.payload())?;
    let m = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    if w.identity()?.digest() != workload.manifest.logical_identity
        || Digest::from_bytes(*m.canonical_digest.bytes()) != mesh.manifest.logical_identity
    {
        return Err(invalid("fluid original source identity differs"));
    }
    let result = FluidSteadySolverResult::decode_owned_on_source(
        &source.payload(),
        source.manifest.payload_digest,
        w,
        &m,
    )?;
    if result.result().identity()?.digest() != source.manifest.logical_identity {
        return Err(invalid("fluid retained result logical identity differs"));
    }
    Ok((m, result))
}

pub fn import_fluid_recovered_fields(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    prepared: &PreparedFluidFieldRecovery,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    response.validate_against(&prepared.program.request)?;
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
        || prepared.inputs.iter().map(host_input).collect::<Vec<_>>()
            != prepared.host.workload.inputs
    {
        return Err(invalid("fluid recovery task or immutable closure differs"));
    }
    require_fluid_recovery_source_decode_budget(&prepared.inputs, &prepared.resource_plan, limits)?;
    let expected = build_prepared_fluid_field_recovery_resource_plan(
        &prepared.request,
        &prepared.inputs,
        prepared.output_policy,
    )?;
    if expected != prepared.resource_plan {
        return Err(invalid("fluid recovery resource policy differs"));
    }
    let w = FluidSteadyWorkload::from_canonical_bytes(
        &unique(
            &prepared.inputs,
            AnalysisNumericalObjectKind::FluidSteadyWorkload,
        )?
        .payload(),
    )?;
    let m = SolverMeshArtifact::canonical_decode(
        &unique(&prepared.inputs, AnalysisNumericalObjectKind::Mesh)?.payload(),
    )
    .map_err(invalid)?;
    let numerical = NumericalPlanIdentity::from_fluid_steady_workload(
        w.identity()?,
        Digest::from_bytes(*m.canonical_digest.bytes()),
    )?;
    if prepared.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, expected.identity.digest())?
        || prepared.host.workload.step != w.producer_identity()?
    {
        return Err(invalid("fluid recovery execution lineage differs"));
    }
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::FluidFieldRecovery(demand) =
        &expected.input.demand
    else {
        return Err(invalid(
            "fluid recovery import requires its typed output demand",
        ));
    };
    let mut bounded = limits;
    bounded.max_total_bytes = bounded
        .max_total_bytes
        .min(demand.output.transport_closure_bytes_upper_bound.0);
    crate::field_recovery::import_recovered_fields(
        source,
        response,
        &prepared.request,
        &prepared.host,
        &prepared.program,
        bounded,
    )
}
