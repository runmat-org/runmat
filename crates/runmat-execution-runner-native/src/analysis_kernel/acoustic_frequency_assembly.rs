//! Governed acoustic fan-in over immutable original numerical artifacts.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{
    decode_acoustic_frequency_assembly_entries, prepare_acoustic_frequency_field_objects,
    rederive_acoustic_frequency_assembly_resource_plan,
};
use runmat_analysis_fea::execution::{
    assemble_acoustic_frequency_result, AcousticFrequencyWorkload, AnalysisNumericalObjectKind,
    ExecutionPlanIdentity, NumericalPlanIdentity,
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
    let workload = AcousticFrequencyWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource = FeaResourcePlan::from_canonical_bytes(&resource_input.payload())?;
    let entries = decode_acoustic_frequency_assembly_entries(invocation.inputs)?;
    let mut expected_outputs = vec![
        AnalysisNumericalObjectKind::SolverResult,
        AnalysisNumericalObjectKind::FrequencySamplePublication,
    ];
    expected_outputs.extend(std::iter::repeat_n(
        AnalysisNumericalObjectKind::Field,
        entries.len(),
    ));
    if invocation.inputs.len() != entries.len() + 3
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
    let assembled = assemble_acoustic_frequency_result(workload, &mesh, entries)?;
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
    let mut outputs = vec![
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
    let mut field_bytes = 0_u64;
    for artifact in &assembled.fields {
        let objects = prepare_acoustic_frequency_field_objects(
            artifact,
            invocation.output_policy.chunk_bytes(),
            invocation.output_policy.inventory_limits(),
        )?;
        field_bytes = field_bytes
            .checked_add(objects.manifest.encoded_length)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid("acoustic fan-in field bytes overflow".into())
            })?;
        outputs.push(AnalysisKernelOutput::prepared(objects));
    }
    if field_bytes > demand.fields_output.encoded_payload_bytes_upper_bound.0 {
        return Err(AnalysisBridgeError::Identity(
            "complete acoustic field inventory exceeds admitted payload bound",
        ));
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests;
