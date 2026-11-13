//! Governed publication from an already retained original terminal block.
use crate::*;
use runmat_analysis_core::StateOutput;
use runmat_analysis_fea::{
    execution::{
        nonlinear_solid_state_publication_requests, AnalysisNumericalObjectKind as K,
        AnalysisWorkKind, NonlinearSolidBlockArtifact, NonlinearSolidWorkload,
        SemanticStatePublicationProfileSource, SemanticStatePublicationRequest,
        StatePublicationProfileEntry, StatePublicationProfiles,
    },
    planning::resource::{
        build_state_publication_resource_plan_with_ceiling, FeaResourceDemand, FeaResourcePlan,
        PreparedPhaseResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{object::validate_inventory, ObjectNamespace};
use std::collections::BTreeSet;

pub struct NonlinearSolidStatePublicationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub block: PreparedAnalysisObjects,
    pub declaration: StateOutput,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
fn unique(
    inputs: &[PreparedAnalysisObjects],
    kind: K,
) -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
    let mut found = inputs.iter().filter(|i| i.manifest.kind == kind);
    match (found.next(), found.next()) {
        (Some(i), None) => Ok(i),
        _ => Err(invalid("nonlinear publication source missing/repeated")),
    }
}
pub fn nonlinear_state_host_input(i: &PreparedAnalysisObjects) -> AnalysisWorkloadInput {
    AnalysisWorkloadInput {
        kind: i.manifest.kind,
        logical_identity: i.manifest.logical_identity,
        payload_digest: i.manifest.payload_digest,
        manifest_digest: i.root.digest,
        manifest_encoded_length: i.root.encoded_length,
        reference_kind: if i.manifest.kind == K::NonlinearSolidBlockArtifact {
            ValueRefKind::ResultObject
        } else {
            ValueRefKind::DriverObject
        },
    }
}
fn authenticate(
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<()> {
    for input in inputs {
        validate_inventory(&input.objects, policy.inventory_limits())?;
        crate::portable_source_admission::authenticate_source_objects(input, input.manifest.kind)?;
    }
    Ok(())
}
fn decode_source(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<(NonlinearSolidWorkload, NonlinearSolidBlockArtifact)> {
    let w = unique(inputs, K::NonlinearSolidWorkload)?;
    let b = unique(inputs, K::NonlinearSolidBlockArtifact)?;
    let workload = NonlinearSolidWorkload::from_canonical_bytes(&w.payload())?;
    // Actual shared profile is derived before the full block join/serde decode.
    let facts = prepared_input_resource_facts_without_resource_plan(inputs)?;
    if facts.decoded_resident_bytes_admission_budget.0
        > workload.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget()
    {
        return Err(invalid(
            "nonlinear publication original decode exceeds authored ceiling",
        ));
    }
    let block = NonlinearSolidBlockArtifact::from_canonical_bytes(
        &b.payload(),
        b.manifest.logical_identity,
    )?;
    if w.manifest.logical_identity != workload.identity()?.digest()
        || block.request.workload != workload.identity()?
        || !block.complete
    {
        return Err(invalid(
            "nonlinear publication original workload/block differs or is partial",
        ));
    }
    Ok((workload, block))
}
pub fn require_nonlinear_state_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<()> {
    plan.validate_identity()?;
    let FeaResourceDemand::SemanticStatePublication(d) = &plan.input.demand else {
        return Err(invalid(
            "nonlinear publication requires shared state resource authority",
        ));
    };
    if d.source != runmat_analysis_fea::execution::SemanticStatePublicationSource::NonlinearSolidOriginalState || inputs.len() != 5 {
        return Err(invalid("nonlinear publication immutable source kinds differ"));
    }
    authenticate(inputs, policy)?;
    for kind in [
        K::NonlinearSolidWorkload,
        K::NonlinearSolidBlockArtifact,
        K::StatePublicationRequest,
        K::StatePublicationProfiles,
        K::ResourcePlan,
    ] {
        unique(inputs, kind)?;
    }
    if decode_admitted_resource_plan(unique(inputs, K::ResourcePlan)?, policy.inventory_limits())?
        != *plan
        || prepared_input_resource_facts_without_resource_plan(inputs)? != d.inputs
        || unique(inputs, K::StatePublicationRequest)?
            .manifest
            .logical_identity
            != d.request
        || unique(inputs, K::NonlinearSolidBlockArtifact)?
            .manifest
            .payload_digest
            != d.source_payload_digest
    {
        return Err(invalid(
            "nonlinear publication actual decode profiles or identities differ",
        ));
    }
    Ok(())
}
pub fn build_prepared_nonlinear_state_publication_resource_plan(
    request: &SemanticStatePublicationRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    if !(4..=5).contains(&inputs.len()) {
        return Err(invalid(
            "nonlinear publication exact source inventory differs",
        ));
    }
    authenticate(inputs, policy)?;
    if inputs.len() == 5 {
        let carried = decode_admitted_resource_plan(
            unique(inputs, K::ResourcePlan)?,
            policy.inventory_limits(),
        )?;
        require_nonlinear_state_decode_budget(inputs, &carried, policy)?;
    }
    let (workload, block) = decode_source(inputs)?;
    let b = unique(inputs, K::NonlinearSolidBlockArtifact)?;
    let c = unique(inputs, K::StatePublicationProfiles)?;
    let r = unique(inputs, K::StatePublicationRequest)?;
    let catalog = StatePublicationProfiles::from_canonical_bytes(&c.payload())?;
    let profile = request.encoding_profile(
        SemanticStatePublicationProfileSource::NonlinearSolidOriginalState {
            workload: &workload,
            block: &block,
        },
    )?;
    if r.payload() != request.canonical_bytes()?
        || r.manifest.logical_identity != Digest::sha256(request.canonical_bytes()?)
        || c.manifest.logical_identity != catalog.identity()
        || catalog.entry(request)?.profile() != profile
    {
        return Err(invalid(
            "nonlinear publication profile/request differs from original source",
        ));
    }
    let plan = build_state_publication_resource_plan_with_ceiling(
        request,
        &catalog,
        b.manifest.payload_digest,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output:
                runmat_analysis_fea::planning::resource::profiled_canonical_output_resource_facts(
                    profile,
                )?,
            output_set: projected_output_set_resource_facts(),
        },
        workload.model.sequence.steps[0].accuracy.resources,
    )?;
    Ok(plan)
}
pub fn prepare_nonlinear_solid_state_publication(
    p: NonlinearSolidStatePublicationPreparation,
) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
    p.source_host.validate()?;
    if p.source_host.workload.kind != AnalysisWorkKind::SolveNonlinearSolidBlock
        || !p.source_host.workload.inputs.iter().any(|i| {
            i.kind == K::NonlinearSolidWorkload
                && i.logical_identity == p.workload.manifest.logical_identity
                && i.payload_digest == p.workload.manifest.payload_digest
        })
    {
        return Err(invalid(
            "nonlinear publication original host/source differs",
        ));
    }
    let mut inputs = vec![p.workload, p.block];
    authenticate(&inputs, p.output_policy)?;
    let (workload, block) = decode_source(&inputs)?;
    let original_request = block.request.identity()?;
    if workload.producer_identity()? != p.source_host.workload.step
        || !p.source_host.workload.inputs.iter().any(|i| {
            i.kind == K::NonlinearSolidBlockRequest && i.logical_identity == original_request
        })
    {
        return Err(invalid(
            "nonlinear publication original producer/request differs",
        ));
    }
    let request = nonlinear_solid_state_publication_requests(&workload, &block)?
        .into_iter()
        .find(|r| r.declaration == p.declaration)
        .ok_or_else(|| invalid("nonlinear publication lacks original declaration"))?;
    let profile = request.encoding_profile(
        SemanticStatePublicationProfileSource::NonlinearSolidOriginalState {
            workload: &workload,
            block: &block,
        },
    )?;
    let catalog = StatePublicationProfiles::new(
        request.producer,
        request.source_result,
        request.topology_digest,
        inputs[1].manifest.payload_digest,
        vec![StatePublicationProfileEntry::new(request.clone(), profile)?],
    )?;
    for (kind, identity, bytes) in [
        (
            K::StatePublicationRequest,
            Digest::sha256(request.canonical_bytes()?),
            request.canonical_bytes()?,
        ),
        (
            K::StatePublicationProfiles,
            catalog.identity(),
            catalog.canonical_bytes()?,
        ),
    ] {
        inputs.push(prepare_analysis_objects(
            kind,
            identity,
            &bytes,
            p.output_policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            p.output_policy.inventory_limits(),
        )?);
    }
    let resource_plan = build_prepared_nonlinear_state_publication_resource_plan(
        &request,
        &inputs,
        p.output_policy,
    )?;
    inputs.push(prepare_analysis_objects(
        K::ResourcePlan,
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
            p.source_host.workload.execution_plan,
            AnalysisWorkKind::PublishState,
            inputs.iter().map(nonlinear_state_host_input).collect(),
            vec![K::SemanticState],
            BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: ANALYSIS_HOST_ABI.into(),
            }]),
        )?,
        p.source_host.artifact_access,
        p.output_policy,
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

pub fn decode_nonlinear_state_publication(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<(NonlinearSolidWorkload, NonlinearSolidBlockArtifact)> {
    require_nonlinear_state_decode_budget(inputs, plan, policy)?;
    decode_source(inputs)
}
