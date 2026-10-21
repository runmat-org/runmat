//! Native original-history load blocks; partial artifacts are not completed studies.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{
    decode_admitted_resource_plan, prepare_analysis_objects,
    rederive_nonlinear_solid_block_resource_plan, require_nonlinear_solid_source_decode_budget,
    AnalysisWorkloadInput,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, AnalysisWorkKind, ExecutionPlanIdentity,
        NonlinearSolidBlockArtifact, NonlinearSolidBlockRequest,
        NonlinearSolidNumericalReceiptArtifact, NonlinearSolidWorkload, NumericalPlanIdentity,
    },
    planning::resource::FeaResourceDemand,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[cfg(test)]
mod tests;

fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    invocation.host.validate()?;
    if invocation.kind != AnalysisWorkKind::SolveNonlinearSolidBlock
        || invocation.host.workload.kind != invocation.kind
        || invocation.host.workload.outputs
            != [Kind::NonlinearSolidBlockArtifact, Kind::NumericalReceipt]
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
                    reference_kind: if input.manifest.kind == Kind::NonlinearSolidBlockArtifact {
                        ValueRefKind::ResultObject
                    } else {
                        ValueRefKind::DriverObject
                    },
                })
                .collect::<Vec<_>>()
    {
        return Err(invalid(
            "nonlinear invocation differs from immutable closure",
        ));
    }
    let policy = invocation.output_policy;
    let plan = decode_admitted_resource_plan(
        unique_input(&invocation, Kind::ResourcePlan)?,
        policy.inventory_limits(),
    )?;
    require_nonlinear_solid_source_decode_budget(
        invocation.inputs,
        &plan,
        policy.inventory_limits(),
    )?;
    let workload_objects = unique_input(&invocation, Kind::NonlinearSolidWorkload)?;
    let workload = NonlinearSolidWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh =
        SolverMeshArtifact::canonical_decode(&unique_input(&invocation, Kind::Mesh)?.payload())
            .map_err(invalid)?;
    let request = NonlinearSolidBlockRequest::from_canonical_bytes(
        &unique_input(&invocation, Kind::NonlinearSolidBlockRequest)?.payload(),
    )?;
    let previous = invocation
        .inputs
        .iter()
        .find(|input| input.manifest.kind == Kind::NonlinearSolidBlockArtifact)
        .map(|input| {
            NonlinearSolidBlockArtifact::from_canonical_bytes(
                &input.payload(),
                input.manifest.logical_identity,
            )
        })
        .transpose()?;
    if invocation.host.workload.step != workload.producer_identity()?
        || workload_objects.manifest.logical_identity != workload.identity()?.digest()
        || rederive_nonlinear_solid_block_resource_plan(
            &workload,
            &mesh,
            &request,
            previous.as_ref(),
            invocation.inputs,
            policy,
        )? != plan
        || invocation.host.workload.execution_plan
            != ExecutionPlanIdentity::new(
                NumericalPlanIdentity::from_nonlinear_solid_block(&request)?,
                plan.identity.digest(),
            )?
    {
        return Err(invalid(
            "nonlinear block original producer/resource authority differs",
        ));
    }
    let FeaResourceDemand::StructuralNonlinearPipeline(demand) = &plan.input.demand else {
        return Err(invalid("nonlinear resource family differs"));
    };
    let transport = demand
        .block_transport
        .as_ref()
        .ok_or_else(|| invalid("nonlinear block budget missing"))?;
    let result = super::with_admitted_cpu_pool(&plan, || {
        Ok(workload.execute_block(mesh, &request, previous.as_ref())?)
    })?;
    let receipt = NonlinearSolidNumericalReceiptArtifact::new(&workload, &result)?;
    let receipt_bytes = receipt.canonical_bytes()?;
    let bytes = result.canonical_bytes()?;
    if bytes.len() as u64 > transport.output.encoded_payload_bytes_upper_bound.0 {
        return Err(invalid("nonlinear block exceeds admitted wire bound"));
    }
    if receipt_bytes.len() as u64 > transport.receipt_output.encoded_payload_bytes_upper_bound.0 {
        return Err(invalid("nonlinear receipt exceeds admitted wire bound"));
    }
    Ok(vec![
        AnalysisKernelOutput::prepared(prepare_analysis_objects(
            Kind::NonlinearSolidBlockArtifact,
            result.identity,
            &bytes,
            policy.chunk_bytes(),
            ObjectNamespace::ResultValue,
            policy.inventory_limits(),
        )?),
        AnalysisKernelOutput::prepared(prepare_analysis_objects(
            Kind::NumericalReceipt,
            receipt.identity().digest(),
            &receipt_bytes,
            policy.chunk_bytes(),
            ObjectNamespace::ResultValue,
            policy.inventory_limits(),
        )?),
    ])
}
