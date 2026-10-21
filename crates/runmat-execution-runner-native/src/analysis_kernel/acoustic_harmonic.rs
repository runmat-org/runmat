//! Portable acoustic_harmonic execution consumes the same admitted numerical and transport policy.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{
    prepare_analysis_objects, rederive_acoustic_harmonic_resource_plan,
};
use runmat_analysis_fea::execution::{
    AcousticHarmonicWorkload, AcousticNumericalReceiptArtifact, AnalysisNumericalObjectKind,
    ExecutionPlanIdentity, NumericalPlanIdentity, SequenceIdentity, StepIdentity,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
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
            "acoustic_harmonic inputs or outputs differ from the workload contract".into(),
        ));
    }
    let workload = AcousticHarmonicWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if workload_input.manifest.logical_identity != workload.identity()?.digest()
        || mesh_input.manifest.logical_identity
            != Digest::from_bytes(*mesh.canonical_digest.bytes())
        || resource_input.manifest.logical_identity != resource_plan.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic_harmonic inputs differ from their canonical identities",
        ));
    }
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("acoustic_harmonic authored step is absent".into())
        })?;
    let expected_step = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
        step,
    )?;
    if invocation.host.workload.step != expected_step {
        return Err(AnalysisBridgeError::Identity(
            "acoustic_harmonic task names a different authored procedure step",
        ));
    }
    let expected_plan = rederive_acoustic_harmonic_resource_plan(
        &workload,
        &mesh,
        invocation.inputs,
        invocation.output_policy,
    )?;
    if resource_plan != expected_plan {
        return Err(AnalysisBridgeError::Identity(
            "acoustic_harmonic resource plan does not converge from authoritative inputs",
        ));
    }
    let expected_execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_acoustic_workload(
            workload.identity()?,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
        )?,
        resource_plan.identity.digest(),
    )?;
    if invocation.host.workload.execution_plan != expected_execution {
        return Err(AnalysisBridgeError::Identity(
            "acoustic_harmonic task execution identity differs from numerical intent and resource plan",
        ));
    }
    let policy = invocation.output_policy;
    let result = super::with_admitted_cpu_pool(&resource_plan, || Ok(workload.execute(&mesh)?))?;
    let receipt = AcousticNumericalReceiptArtifact::new(&workload, &result, &mesh)?;
    let bytes = result.canonical_bytes()?;
    let bound = runmat_analysis_fea::planning::resource::acoustic_harmonic_result_payload_bytes_upper_bound(&workload, &mesh)?;
    if bytes.len() as u64 > bound {
        return Err(AnalysisBridgeError::Invalid(
            "acoustic result exceeds admitted payload bound".into(),
        ));
    }
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        result.source_result_identity()?.digest(),
        &bytes,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    // The solve publishes metadata for the independently admitted state task. Profiling borrows
    // the completed numerical fields; it does not construct the state or run recovery again.
    let profiles = result.state_publication_profiles(Digest::sha256(&bytes))?;
    let profile_bytes = profiles.canonical_bytes()?;
    let profile_bound = workload
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ))?
        .encoded_bytes_upper_bound;
    if profile_bytes.len() as u64 > profile_bound {
        return Err(AnalysisBridgeError::Invalid(
            "acoustic state profile catalog exceeds admitted payload bound".into(),
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

#[cfg(test)]
mod tests;
