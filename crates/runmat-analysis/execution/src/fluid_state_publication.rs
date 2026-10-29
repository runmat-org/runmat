//! Separate complete FV publication from the immutable original solve closure.
use crate::portable_source_admission::authenticate_source_objects;
use crate::*;
use runmat_analysis_core::StateOutput;
use runmat_analysis_fea::{
    execution::{
        fluid_state_publication_profile, fluid_state_publication_requests,
        AnalysisNumericalObjectKind as K, AnalysisWorkKind, FluidSteadySolverResult,
        FluidSteadyWorkload, SemanticStatePublicationRequest,
    },
    planning::resource::{
        build_fluid_state_publication_resource_plan, FeaResourceDemand, FeaResourcePlan,
        PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{
    object::{validate_inventory, ObjectInventoryLimits},
    ObjectNamespace,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use std::collections::BTreeSet;

pub struct FluidStatePublicationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_inputs: Vec<PreparedAnalysisObjects>,
    pub source_result: PreparedAnalysisObjects,
    pub declaration: StateOutput,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
fn unique(
    inputs: &[PreparedAnalysisObjects],
    kind: K,
) -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
    let mut found = inputs.iter().filter(|i| i.manifest.kind == kind);
    let value = found
        .next()
        .ok_or_else(|| invalid("fluid state source absent"))?;
    if found.next().is_some() {
        return Err(invalid("fluid state source repeated"));
    }
    Ok(value)
}
fn host_input(i: &PreparedAnalysisObjects) -> AnalysisWorkloadInput {
    AnalysisWorkloadInput {
        kind: i.manifest.kind,
        logical_identity: i.manifest.logical_identity,
        payload_digest: i.manifest.payload_digest,
        manifest_digest: i.root.digest,
        manifest_encoded_length: i.root.encoded_length,
        reference_kind: if i.manifest.kind == K::SolverResult {
            ValueRefKind::ResultObject
        } else {
            ValueRefKind::DriverObject
        },
    }
}
fn source_plan(
    inputs: &[PreparedAnalysisObjects],
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<(&PreparedAnalysisObjects, FeaResourcePlan)> {
    let mut selected = None;
    for i in inputs.iter().filter(|i| i.manifest.kind == K::ResourcePlan) {
        let plan = decode_admitted_resource_plan(i, limits)?;
        if matches!(plan.input.demand, FeaResourceDemand::FluidSteadySolve(_)) {
            if selected.is_some() {
                return Err(invalid("fluid state repeats original solve plan"));
            }
            selected = Some((i, plan));
        }
    }
    selected.ok_or_else(|| invalid("fluid state lacks original solve resource authority"))
}
fn decode_source(
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<(
    FluidSteadyWorkload,
    SolverMeshArtifact,
    FluidSteadySolverResult,
)> {
    let (plan_input, plan) = source_plan(inputs, policy.inventory_limits())?;
    let FeaResourceDemand::FluidSteadySolve(demand) = &plan.input.demand else {
        unreachable!()
    };
    let w = unique(inputs, K::FluidSteadyWorkload)?;
    let m = unique(inputs, K::Mesh)?;
    let r = unique(inputs, K::SolverResult)?;
    for i in [w, m, r] {
        validate_inventory(&i.objects, policy.inventory_limits())?;
        authenticate_source_objects(i, i.manifest.kind)?;
    }
    let wire = w
        .manifest
        .encoded_length
        .checked_add(m.manifest.encoded_length)
        .ok_or_else(|| invalid("fluid source wire overflow"))?;
    if wire != demand.inputs.encoded_payload_bytes.0
        || wire > demand.inputs.decoded_resident_bytes_admission_budget.0
        || r.manifest.encoded_length > demand.result_output.encoded_payload_bytes_upper_bound.0
    {
        return Err(invalid(
            "fluid state source exceeds original admitted wire bounds",
        ));
    }
    // Preserve the original solve ordering, and borrow its closure rather than clone
    // chunk buffers while the full original transport remains live.
    if inputs.len() < 3 || &inputs[0] != w || &inputs[1] != m || &inputs[2] != plan_input {
        return Err(invalid(
            "fluid publication must preserve the original solve input prefix",
        ));
    }
    let original_facts =
        crate::object::prepared_input_resource_facts_from_refs([w, m].into_iter())?;
    let result_facts = crate::object::prepared_input_resource_facts_from_refs(std::iter::once(r))?;
    if original_facts != demand.inputs
        || result_facts.decoded_resident_bytes_admission_budget.0
            > demand
                .result_output
                .construction_resident_bytes_admission_budget
                .0
    {
        return Err(invalid(
            "fluid original source actual codec profile exceeds solve admission",
        ));
    }
    let workload = FluidSteadyWorkload::from_canonical_bytes(&w.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&m.payload()).map_err(invalid)?;
    if rederive_fluid_steady_resource_plan(&workload, &mesh, &inputs[..3], policy)? != plan {
        return Err(invalid("fluid original solve authority differs"));
    }
    let result = FluidSteadySolverResult::from_canonical_bytes_for(
        &r.payload(),
        r.manifest.payload_digest,
        &workload,
        &mesh,
    )?;
    if result.identity()?.digest() != r.manifest.logical_identity {
        return Err(invalid("fluid original result identity differs"));
    }
    Ok((workload, mesh, result))
}

/// Memoized per exact request, input content and policy: every stage of a study asks for the
/// same plan, and each derivation decodes and re-prepares the original fluid source.
pub fn build_prepared_fluid_state_publication_resource_plan(
    request: &SemanticStatePublicationRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    build_plan_with_source(request, inputs, policy, None)
}

type DecodedSource = (
    FluidSteadyWorkload,
    SolverMeshArtifact,
    FluidSteadySolverResult,
);

/// `decoded` is the source `decode_source` already authenticated from these same original
/// inputs (workload, mesh, result and solve plan); it is not decoded a second time.
fn build_plan_with_source(
    request: &SemanticStatePublicationRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    decoded: Option<&DecodedSource>,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    static MEMO: crate::fluid_phase_plan_memo::PlanMemo =
        crate::fluid_phase_plan_memo::PlanMemo::new();
    MEMO.get_or_derive(
        "fluid-state-publication",
        &request.canonical_bytes()?,
        inputs,
        policy,
        || derive_prepared_fluid_state_publication_resource_plan(request, inputs, policy, decoded),
    )
}

fn derive_prepared_fluid_state_publication_resource_plan(
    request: &SemanticStatePublicationRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    decoded: Option<&DecodedSource>,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let (source_plan_input, _) = source_plan(inputs, policy.inventory_limits())?;
    if !(5..=6).contains(&inputs.len())
        || inputs[2].manifest.logical_identity != source_plan_input.manifest.logical_identity
    {
        return Err(invalid(
            "fluid state requires exact original workload/mesh/result/solve-plan/request",
        ));
    }
    let numerical = &inputs[..5];
    for kind in [
        K::FluidSteadyWorkload,
        K::Mesh,
        K::SolverResult,
        K::StatePublicationRequest,
        K::ResourcePlan,
    ] {
        unique(numerical, kind)?;
    }
    if inputs.len() == 6 {
        if inputs[5].manifest.kind != K::ResourcePlan {
            return Err(invalid(
                "fluid publication sixth input must be its resource authority",
            ));
        }
        let carried = decode_admitted_resource_plan(&inputs[5], policy.inventory_limits())?;
        require_fluid_state_source_decode_budget(inputs, &carried, policy.inventory_limits())?;
    }
    let req = unique(numerical, K::StatePublicationRequest)?;
    authenticate_source_objects(req, K::StatePublicationRequest)?;
    if req.manifest.logical_identity != Digest::sha256(request.canonical_bytes()?)
        || req.payload() != request.canonical_bytes()?
    {
        return Err(invalid("fluid state request differs"));
    }
    let owned;
    let (w, m, r) = match decoded {
        Some(source) => source,
        None => {
            owned = decode_source(numerical, policy)?;
            &owned
        }
    };
    let source = unique(numerical, K::SolverResult)?;
    let catalog =
        fluid_state_publication_profile(request, w, m, r, source.manifest.payload_digest)?;
    let profile = catalog.entry(request)?.profile();
    Ok(build_fluid_state_publication_resource_plan(
        request,
        w,
        m,
        r,
        source.manifest.payload_digest,
        &catalog,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts(numerical)?,
            output:
                runmat_analysis_fea::planning::resource::profiled_canonical_output_resource_facts(
                    profile,
                )?,
            output_set: projected_output_set_resource_facts(),
        },
    )?)
}

pub fn require_fluid_state_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    plan.validate_identity()?;
    let FeaResourceDemand::SemanticStatePublication(demand) = &plan.input.demand else {
        return Err(invalid("fluid state requires shared publication phase"));
    };
    if demand.source
        != runmat_analysis_fea::execution::SemanticStatePublicationSource::FluidPrimaryFields
    {
        return Err(invalid("fluid publication source differs"));
    }
    let (original, _) = source_plan(inputs, limits)?;
    if !(5..=6).contains(&inputs.len()) || inputs[2] != *original {
        return Err(invalid("fluid publication original solve prefix differs"));
    }
    for kind in [
        K::FluidSteadyWorkload,
        K::Mesh,
        K::SolverResult,
        K::StatePublicationRequest,
    ] {
        unique(inputs, kind)?;
    }
    let mut wire = 0u64;
    for i in inputs {
        validate_inventory(&i.objects, limits)?;
        authenticate_source_objects(i, i.manifest.kind)?;
        if i.manifest.kind == K::ResourcePlan
            && i.manifest.logical_identity != original.manifest.logical_identity
        {
            if decode_admitted_resource_plan(i, limits)? != *plan {
                return Err(invalid("fluid publication plan differs"));
            }
        } else {
            wire = wire
                .checked_add(i.manifest.encoded_length)
                .ok_or_else(|| invalid("fluid publication wire overflow"))?;
        }
    }
    if wire != demand.inputs.encoded_payload_bytes.0
        || wire > demand.inputs.decoded_resident_bytes_admission_budget.0
    {
        return Err(invalid(
            "fluid publication exceeds carried decode admission",
        ));
    }
    // Borrowed wire counts authenticate first; only then may the shared actual-codec
    // profiler join bounded partition bytes. A self-hashed lowered budget is not authority.
    let facts = prepared_input_resource_facts(&inputs[..5])?;
    if facts != demand.inputs
        || demand.source_validation_workspace_bytes < facts.decoded_resident_bytes_admission_budget
        || unique(inputs, K::StatePublicationRequest)?
            .manifest
            .logical_identity
            != demand.request
        || unique(inputs, K::SolverResult)?.manifest.logical_identity
            != demand.source_result.digest()
    {
        return Err(invalid(
            "fluid publication actual source profiles or identities differ",
        ));
    }
    Ok(())
}

pub fn prepare_fluid_state_publication(
    p: FluidStatePublicationPreparation,
) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveFluidSteady
        || p.source_inputs.len() != 3
        || p.source_inputs.iter().map(host_input).collect::<Vec<_>>()
            != p.source_host.workload.inputs
    {
        return Err(invalid("fluid publication original host closure differs"));
    }
    let policy = AnalysisOutputPolicy::new(p.chunk_bytes, p.inventory_limits)?;
    let mut inputs = p.source_inputs;
    inputs.push(p.source_result);
    let source = decode_source(&inputs, policy)?;
    let (w, m, r) = &source;
    if r.producer_identity() != p.source_host.workload.step {
        return Err(invalid("fluid publication producer differs"));
    }
    let request = fluid_state_publication_requests(w, m, r)?
        .into_iter()
        .find(|r| r.declaration == p.declaration)
        .ok_or_else(|| invalid("fluid state declaration differs from original authored output"))?;
    let bytes = request.canonical_bytes()?;
    inputs.push(prepare_analysis_objects(
        K::StatePublicationRequest,
        Digest::sha256(&bytes),
        &bytes,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.inventory_limits,
    )?);
    let resource_plan = build_plan_with_source(&request, &inputs, policy, Some(&source))?;
    inputs.push(prepare_analysis_objects(
        K::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.inventory_limits,
    )?);
    let workload = AnalysisWorkloadRequest::new(
        p.source_host.workload.study,
        p.source_host.workload.step,
        p.source_host.workload.execution_plan,
        AnalysisWorkKind::PublishState,
        inputs.iter().map(host_input).collect(),
        vec![K::SemanticState],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        workload,
        p.source_host.artifact_access,
        crate::AnalysisOutputPolicy::new(p.chunk_bytes, p.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedSemanticStatePublication {
        request,
        host,
        program,
        inputs,
        resource_plan,
    })
}
