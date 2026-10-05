//! Governed original finite-volume CPU solve; no reduced channel or nodal substitution.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{
    decode_admitted_resource_plan, prepare_analysis_objects,
    project_fluid_transient_checkpoint_output, require_fluid_transient_source_closure,
    require_fluid_transient_source_decode_budget, AnalysisWorkloadInput,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity,
    FluidTransientBlockWorkload, NumericalPlanIdentity,
};
use runmat_analysis_fea::planning::resource::FeaResourceDemand;
use runmat_execution::{value::ValueRefKind, Digest};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[cfg(test)]
pub(crate) mod tests;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{
        FluidTransientBlockWorkload as Workload, Mesh, ResourcePlan,
    };
    invocation.host.validate()?;
    let workload_objects = unique_input(&invocation, Workload)?;
    let mesh_objects = unique_input(&invocation, Mesh)?;
    let plan_objects = unique_input(&invocation, ResourcePlan)?;
    if invocation.kind != AnalysisWorkKind::SolveFluidTransientBlock
        || invocation.host.workload.kind != invocation.kind
        || invocation.inputs.len() != 3
        || invocation.host.workload.outputs
            != [AnalysisNumericalObjectKind::FluidTransientCheckpoint]
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
        return Err(invalid(
            "fluid invocation differs from immutable input/output closure",
        ));
    }
    // Bounded authenticated plan first; original source join/decode follows admission.
    let plan =
        decode_admitted_resource_plan(plan_objects, invocation.output_policy.inventory_limits())?;
    require_fluid_transient_source_decode_budget(
        invocation.inputs,
        &plan,
        invocation.output_policy.inventory_limits(),
    )?;
    let workload = FluidTransientBlockWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload()).map_err(invalid)?;
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    if workload_objects.manifest.logical_identity != workload.identity()?.digest()
        || mesh_objects.manifest.logical_identity != topology
        || plan_objects.manifest.logical_identity != plan.identity.digest()
        || invocation.host.workload.step != workload.producer_identity()?
    {
        return Err(invalid(
            "fluid task differs from original workload/mesh/producer",
        ));
    }
    require_fluid_transient_source_closure(
        &workload,
        &mesh,
        invocation.inputs,
        invocation.output_policy,
    )?;
    let execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_fluid_transient_block_workload(workload.identity()?, topology)?,
        plan.identity.digest(),
    )?;
    if invocation.host.workload.execution_plan != execution {
        return Err(invalid(
            "fluid execution identity differs from physical intent/resource plan",
        ));
    }
    let FeaResourceDemand::FluidTransientBlock(demand) = &plan.input.demand else {
        return Err(invalid(
            "fluid solve requires its own typed resource demand",
        ));
    };
    let prepared = workload.prepare_with_resource_plan(&mesh, &plan)?;
    let profile = prepared.result_encoding_profile()?;
    let actual_output = project_fluid_transient_checkpoint_output(
        profile.encoded_bytes_upper_bound,
        runmat_analysis_fea::planning::resource::ByteCount(
            profile.materialization_resident_bytes_estimate,
        ),
        invocation.output_policy,
    )?;
    if actual_output != demand.output {
        return Err(invalid(
            "fluid transient output policy differs from admission",
        ));
    }
    let result = prepared.execute()?;
    let result_bytes = result.canonical_bytes()?;
    if result_bytes.len() as u64 > demand.output.encoded_payload_bytes_upper_bound.0 {
        return Err(invalid(
            "fluid transient checkpoint exceeds admitted payload",
        ));
    }
    let policy = invocation.output_policy;
    let output = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FluidTransientCheckpoint,
        result.identity()?,
        &result_bytes,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    Ok(vec![AnalysisKernelOutput::prepared(output)])
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
