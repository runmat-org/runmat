//! Static magnetic work executes only after exact input and allocation-plan rederivation.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{prepare_analysis_objects, rederive_magnetostatic_resource_plan};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, ExecutionPlanIdentity, MagneticNumericalReceiptArtifact,
    MagnetostaticWorkload, NumericalPlanIdentity,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[cfg(test)]
mod tests;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::MagnetostaticWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len() != 3
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "magnetostatic inputs or outputs differ from the task contract".into(),
        ));
    }
    let workload = MagnetostaticWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_input.payload())?;
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    if workload_input.manifest.logical_identity != workload.identity()?.digest()
        || mesh_input.manifest.logical_identity != topology
        || resource_input.manifest.logical_identity != resource_plan.identity.digest()
        || invocation.host.workload.step != workload.producer_identity()?
    {
        return Err(AnalysisBridgeError::Identity(
            "magnetostatic task differs from canonical input or producer identities",
        ));
    }
    let expected_plan = rederive_magnetostatic_resource_plan(
        &workload,
        &mesh,
        invocation.inputs,
        invocation.output_policy,
    )?;
    if resource_plan != expected_plan {
        return Err(AnalysisBridgeError::Identity(
            "magnetostatic resource plan differs from authoritative inputs",
        ));
    }
    let expected_execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_magnetostatic_workload(workload.identity()?, topology)?,
        resource_plan.identity.digest(),
    )?;
    if invocation.host.workload.execution_plan != expected_execution {
        return Err(AnalysisBridgeError::Identity(
            "magnetostatic execution identity differs from numerical intent and allocation",
        ));
    }
    let result = super::with_admitted_cpu_pool(&resource_plan, || {
        Ok(workload.execute_with_resource_plan(&mesh, &resource_plan)?)
    })?;
    let receipt = MagneticNumericalReceiptArtifact::new(&workload, &result, &mesh)?;
    let bytes = result.canonical_bytes()?;
    let bound =
        runmat_analysis_fea::planning::resource::magnetostatic_result_payload_bytes_upper_bound(
            &workload, &mesh,
        )?;
    if bytes.len() as u64 > bound {
        return Err(AnalysisBridgeError::Invalid(
            "magnetostatic result exceeds admitted payload bound".into(),
        ));
    }
    let policy = invocation.output_policy;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        result.source_result_identity()?.digest(),
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
            "magnetostatic profiles exceed admitted payload bound".into(),
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
