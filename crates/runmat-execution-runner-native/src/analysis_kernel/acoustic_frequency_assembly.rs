//! Governed acoustic fan-in over immutable original numerical artifacts.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{
    decode_acoustic_frequency_assembly_entries, rederive_acoustic_frequency_assembly_resource_plan,
};
use runmat_analysis_fea::execution::{
    assemble_acoustic_frequency_result, AcousticFrequencyAssemblyRequest,
    AcousticFrequencyWorkload, AnalysisNumericalObjectKind, ExecutionPlanIdentity,
    NumericalPlanIdentity,
};
use runmat_analysis_fea::planning::resource::{FeaResourceDemand, FeaResourcePlan};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::AcousticFrequencyWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::AcousticFrequencyAssemblyRequest,
    )?;
    let request = AcousticFrequencyAssemblyRequest::from_canonical_bytes(&request_input.payload())?;
    let workload = AcousticFrequencyWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource = FeaResourcePlan::from_canonical_bytes(&resource_input.payload())?;
    let entries = decode_acoustic_frequency_assembly_entries(invocation.inputs)?;
    let expected_outputs = vec![
        AnalysisNumericalObjectKind::SolverResult,
        AnalysisNumericalObjectKind::FrequencySamplePublication,
    ];
    if invocation.inputs.len() != entries.len() + 4
        || request_input.manifest.logical_identity != request.identity()
        || invocation.host.workload.outputs != expected_outputs
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic fan-in has foreign inputs or outputs",
        ));
    }
    let expected_resource = rederive_acoustic_frequency_assembly_resource_plan(
        &workload,
        &mesh,
        &entries,
        invocation.inputs,
    )?;
    let producer = entries[0].producer();
    let expected_execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_acoustic_frequency_workload(
            workload.identity()?,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
        )?,
        expected_resource.identity.digest(),
    )?;
    if workload_input.manifest.logical_identity != workload.identity()?.digest()
        || mesh_input.manifest.logical_identity
            != Digest::from_bytes(*mesh.canonical_digest.bytes())
        || resource_input.manifest.logical_identity != resource.identity.digest()
        || resource != expected_resource
        || invocation.host.workload.step != producer
        || invocation.host.workload.execution_plan != expected_execution
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic fan-in resource or numerical authority differs",
        ));
    }
    let assembled = assemble_acoustic_frequency_result(workload, &mesh, entries, &request)?;
    let FeaResourceDemand::AcousticFrequencyAssembly(demand) = &resource.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "acoustic fan-in requires its exact resource demand",
        ));
    };
    let result_bytes = assembled.result.canonical_bytes()?;
    let publication_bytes = assembled.publication.canonical_bytes()?;
    if result_bytes.len() as u64 > demand.result_output.encoded_payload_bytes_upper_bound.0
        || publication_bytes.len() as u64
            > demand
                .publication_output
                .encoded_payload_bytes_upper_bound
                .0
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic fan-in metadata exceeds admitted payload bounds",
        ));
    }
    let outputs = vec![
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::SolverResult,
            assembled.result.source_result_identity()?.digest(),
            result_bytes,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::FrequencySamplePublication,
            assembled.publication.identity().digest(),
            publication_bytes,
        ),
    ];
    Ok(outputs)
}

#[cfg(test)]
mod tests;
