use super::*;

pub(super) fn execute_scope_preparation(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralTransientWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let modal_input =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len() != 3 + usize::from(modal_input.is_some())
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::StructuralTransientCheckpointScope]
    {
        return Err(AnalysisBridgeError::Invalid(
            "transient scope task inputs or output differ from its workload contract".into(),
        ));
    }
    let workload = StructuralTransientWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let modal_state = modal_input
        .map(decode_prepared_semantic_state)
        .transpose()?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if workload_input.manifest.logical_identity
        != runmat_execution::Digest::sha256(workload.canonical_bytes()?)
        || mesh_input.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        || modal_input
            .zip(modal_state.as_ref())
            .is_some_and(|(input, state)| {
                input.manifest.logical_identity != state.identity.digest()
            })
        || resource_input.manifest.logical_identity != resource_plan.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "transient scope task inputs differ from their canonical identities",
        ));
    }
    let expected_plan =
        runmat_analysis_fea::pipeline::transient::plan_transient_checkpoint_scope_task(
            &workload,
            &mesh,
            modal_state.as_ref(),
            prepared_input_resource_facts_without_resource_plan(invocation.inputs)?,
            projected_output_set_resource_facts(),
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                resource_plan.request.cpu_millicores,
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let (study, step, execution_plan) =
        runmat_analysis_execution::transient_scope_task_identity(&workload, &mesh, &resource_plan)?;
    if resource_plan != expected_plan
        || invocation.host.workload.study != study
        || invocation.host.workload.step != step
        || invocation.host.workload.execution_plan != execution_plan
        || !resource_plan.request.accelerators.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "transient scope task plan or authored identity does not converge on the worker",
        ));
    }
    let scope = super::with_admitted_cpu_pool(&resource_plan, || {
        runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
            &workload,
            &mesh,
            modal_state.as_ref(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
    })?;
    if scope.study != study || scope.step != step {
        return Err(AnalysisBridgeError::Identity(
            "constructed transient scope differs from its admitted study or step",
        ));
    }
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::StructuralTransientCheckpointScope,
        scope.identity()?.digest(),
        scope.canonical_bytes()?,
    )])
}
