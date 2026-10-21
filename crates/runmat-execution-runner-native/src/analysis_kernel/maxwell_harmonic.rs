//! Exact governed complex Maxwell execution; no reduced graph or source reinterpretation.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{
    decode_admitted_resource_plan, prepare_analysis_objects,
    rederive_maxwell_harmonic_resource_plan, require_maxwell_source_decode_budget,
    AnalysisWorkloadInput,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity, MaxwellHarmonicWorkload,
    MaxwellNumericalReceiptArtifact, NumericalPlanIdentity,
};
use runmat_analysis_fea::planning::resource::maxwell_harmonic_result_payload_bytes_upper_bound;
use runmat_execution::{value::ValueRefKind, Digest};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[cfg(test)]
pub(crate) mod tests;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    invocation.host.validate()?;
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.kind != AnalysisWorkKind::SolveMaxwellHarmonic
        || invocation.host.workload.kind != invocation.kind
        || invocation.inputs.len() != 3
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
        || invocation.host.workload.inputs
            != invocation
                .inputs
                .iter()
                .map(|input| AnalysisWorkloadInput {
                    kind: input.manifest.kind,
                    logical_identity: input.manifest.logical_identity,
                    payload_digest: input.manifest.payload_digest,
                    manifest_digest: input.root.digest,
                    manifest_encoded_length: input.root.encoded_length,
                    reference_kind: ValueRefKind::DriverObject,
                })
                .collect::<Vec<_>>()
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell invocation differs from immutable task input/output closure",
        ));
    }
    // Authenticate and decode only the bounded plan first, then admit original source bytes.
    let plan =
        decode_admitted_resource_plan(resource_input, invocation.output_policy.inventory_limits())?;
    require_maxwell_source_decode_budget(
        invocation.inputs,
        &plan,
        invocation.output_policy.inventory_limits(),
    )?;
    let workload = MaxwellHarmonicWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    if workload_input.manifest.logical_identity != workload.identity()?.digest()
        || mesh_input.manifest.logical_identity != topology
        || resource_input.manifest.logical_identity != plan.identity.digest()
        || invocation.host.workload.step != workload.producer_identity()?
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell task differs from canonical source/mesh/producer identities",
        ));
    }
    let expected = rederive_maxwell_harmonic_resource_plan(
        &workload,
        &mesh,
        invocation.inputs,
        invocation.output_policy,
    )?;
    if expected != plan {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell plan differs from original input and exact complex support",
        ));
    }
    let execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_maxwell_harmonic_workload(workload.identity()?, topology)?,
        plan.identity.digest(),
    )?;
    if invocation.host.workload.execution_plan != execution {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell execution identity differs from physical intent and resource allocation",
        ));
    }
    // The admitted path rederives allocation before quadrature, assembly, or shared GMRES.
    let result = super::with_admitted_cpu_pool(&plan, || {
        Ok(workload.execute_with_resource_plan(&mesh, &plan)?)
    })?;
    let receipt = MaxwellNumericalReceiptArtifact::new(&workload, &result, &mesh)?;
    let bytes = result.canonical_bytes()?;
    if bytes.len() as u64 > maxwell_harmonic_result_payload_bytes_upper_bound(&workload, &mesh)? {
        return Err(AnalysisBridgeError::Invalid(
            "Maxwell result exceeds admitted payload bound".into(),
        ));
    }
    let policy = invocation.output_policy;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        result.identity()?.digest(),
        &bytes,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    let profiles = result.state_publication_profiles(Digest::sha256(&bytes))?;
    let profile_bytes = profiles.canonical_bytes()?;
    if profile_bytes.len() as u64
        > workload
            .state_publication_profiles_encoding_profile(topology)?
            .encoded_bytes_upper_bound
    {
        return Err(AnalysisBridgeError::Invalid(
            "Maxwell state profiles exceed admitted payload bound".into(),
        ));
    }
    let profile_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles.identity(),
        &profile_bytes,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    Ok(vec![
        AnalysisKernelOutput::prepared(objects),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::NumericalReceipt,
            receipt.identity().digest(),
            receipt.canonical_bytes()?,
        ),
        AnalysisKernelOutput::prepared(profile_objects),
    ])
}
