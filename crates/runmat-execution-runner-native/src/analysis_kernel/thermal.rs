//! Portable thermal execution consumes the same admitted numerical and transport policy.
use super::{
    unique_input, AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_execution::{
    execute_thermal_solver_result_objects, prepare_analysis_objects,
    rederive_thermal_resource_plan, thermal_receipt_from_prepared_result,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, ExecutionPlanIdentity, NumericalPlanIdentity, SequenceIdentity,
    StepIdentity, ThermalWorkload,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(&invocation, AnalysisNumericalObjectKind::ThermalWorkload)?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len() != 3
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "thermal inputs or outputs differ from the workload contract".into(),
        ));
    }
    let workload = ThermalWorkload::from_canonical_bytes(&workload_input.payload())?;
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
            "thermal inputs differ from their canonical identities",
        ));
    }
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal authored step is absent".into()))?;
    let expected_step = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
        step,
    )?;
    if invocation.host.workload.step != expected_step {
        return Err(AnalysisBridgeError::Identity(
            "thermal task names a different authored procedure step",
        ));
    }
    let expected_plan = rederive_thermal_resource_plan(
        &workload,
        &mesh,
        invocation.inputs,
        invocation.output_policy,
    )?;
    if resource_plan != expected_plan {
        return Err(AnalysisBridgeError::Identity(
            "thermal resource plan does not converge from authoritative inputs",
        ));
    }
    let expected_execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_thermal_intent(
            workload.numerical_intent_identity()?,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
        )?,
        resource_plan.identity.digest(),
    )?;
    if invocation.host.workload.execution_plan != expected_execution {
        return Err(AnalysisBridgeError::Identity(
            "thermal task execution identity differs from numerical intent and resource plan",
        ));
    }
    let policy = invocation.output_policy;
    let result = execute_thermal_solver_result_objects(
        &workload,
        &mesh,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    let receipt = thermal_receipt_from_prepared_result(&workload, &result)?;
    let receipt_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt.identity().digest(),
        &receipt.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    Ok(vec![
        AnalysisKernelOutput::prepared(result),
        AnalysisKernelOutput::prepared(receipt_objects),
    ])
}

#[cfg(test)]
mod tests;
