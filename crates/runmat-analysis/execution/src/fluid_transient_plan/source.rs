//! One source/admission owner for workers and cold checkpoint consumers.
use super::*;
use crate::{AnalysisHostWorkload, AnalysisWorkloadInput};
use runmat_analysis_fea::execution::{
    AnalysisWorkKind, ExecutionPlanIdentity, NumericalPlanIdentity,
    PreparedFluidTransientContinuation,
};
use runmat_execution::value::ValueRefKind;

/// Authenticated source values. Physical preparation below rederives the plan;
/// decoding and hashes alone are not numerical validation.
pub struct DecodedFluidTransientTaskSource {
    pub request: FluidTransientBlockWorkload,
    pub mesh: SolverMeshArtifact,
    pub resource_plan: FeaResourcePlan,
}

impl DecodedFluidTransientTaskSource {
    pub fn prepare(
        &self,
        policy: AnalysisOutputPolicy,
    ) -> AnalysisBridgeResult<PreparedFluidTransientContinuation<'_>> {
        let prepared = self
            .request
            .prepare_with_resource_plan(&self.mesh, &self.resource_plan)?;
        let profile = prepared.result_encoding_profile()?;
        let output = project_fluid_transient_checkpoint_output(
            profile.encoded_bytes_upper_bound,
            runmat_analysis_fea::planning::resource::ByteCount(
                profile.materialization_resident_bytes_estimate,
            ),
            policy,
        )?;
        let facts = self
            .resource_plan
            .input
            .demand
            .output_facts(Kind::FluidTransientCheckpoint)
            .ok_or_else(|| invalid("fluid transient checkpoint admission missing"))?;
        if &output != facts {
            return Err(invalid(
                "fluid transient output policy differs from admission",
            ));
        }
        Ok(prepared)
    }
}

pub fn decode_fluid_transient_task_source(
    host: &AnalysisHostWorkload,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<DecodedFluidTransientTaskSource> {
    host.validate()?;
    if host.workload.kind != AnalysisWorkKind::SolveFluidTransientBlock
        || host.output_policy != policy
        || host.workload.outputs != [Kind::FluidTransientCheckpoint]
        || host.workload.inputs
            != inputs
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
            "fluid transient immutable task source/output closure differs",
        ));
    }
    let input = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut found = inputs.iter().filter(|input| input.manifest.kind == kind);
        let result = found
            .next()
            .ok_or_else(|| invalid("fluid transient source absent"))?;
        if found.next().is_some() {
            return Err(invalid("duplicate fluid transient source"));
        }
        Ok(result)
    };
    // Authenticate the bounded plan before joining/decoding either physical input.
    let resource_plan = crate::decode_admitted_resource_plan(
        input(Kind::ResourcePlan)?,
        policy.inventory_limits(),
    )?;
    crate::require_fluid_transient_source_decode_budget(
        inputs,
        &resource_plan,
        policy.inventory_limits(),
    )?;
    let request = FluidTransientBlockWorkload::from_canonical_bytes(
        &input(Kind::FluidTransientBlockWorkload)?.payload(),
    )?;
    let mesh =
        SolverMeshArtifact::canonical_decode(&input(Kind::Mesh)?.payload()).map_err(invalid)?;
    require_fluid_transient_source_closure(&request, &mesh, inputs, policy)?;
    let execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_fluid_transient_block_workload(
            request.identity()?,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
        )?,
        resource_plan.identity.digest(),
    )?;
    if host.workload.step != request.producer_identity()?
        || host.workload.execution_plan != execution
    {
        return Err(invalid(
            "fluid transient producer or execution identity differs",
        ));
    }
    Ok(DecodedFluidTransientTaskSource {
        request,
        mesh,
        resource_plan,
    })
}
