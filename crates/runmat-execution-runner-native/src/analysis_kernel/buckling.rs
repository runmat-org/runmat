use super::*;

pub(super) fn execute_structural_buckling(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralBucklingWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let state_input = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    let constraint_input = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len() != 5
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural buckling inputs or outputs differ from the workload contract".into(),
        ));
    }
    let workload = decode_prepared_structural_buckling_workload(workload_input)?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let state = decode_prepared_semantic_state(state_input)?;
    let constraints =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&constraint_input.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    constraints
        .validate_against(workload.model(), &mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let topology_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
    let expected_constraint_identity = ConstraintPlanIdentity::new(
        invocation.host.workload.step,
        topology_digest,
        constraints
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    if workload_input.manifest.logical_identity != workload.identity().digest()
        || mesh_input.manifest.logical_identity != topology_digest
        || state_input.manifest.logical_identity != state.identity.digest()
        || constraint_input.manifest.logical_identity != expected_constraint_identity.digest()
        || resource_input.manifest.logical_identity != resource_plan.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "structural buckling inputs differ from their canonical identities",
        ));
    }
    let expected_step = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
            &workload.model().sequence,
        )?,
        workload.procedure_step()?,
    )?;
    if invocation.host.workload.step != expected_step {
        return Err(AnalysisBridgeError::Identity(
            "structural buckling task names a different authored step",
        ));
    }
    let linearization = state
        .payload
        .structural_linearization()
        .ok_or_else(|| AnalysisBridgeError::Invalid("buckling linearization is absent".into()))?;
    let result_bound = structural_buckling_result_payload_bytes_upper_bound(
        linearization.base_displacement().len(),
        workload.solver().requested_mode_count(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_structural_buckling_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &state,
        &resource_plan.request.accelerators,
        PreparedStructuralBucklingResourceFacts {
            local_execution:
                runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                    resource_plan.request.cpu_millicores,
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            inputs: prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            result_output: projected_output_resource_facts(result_bound)?,
            receipt_output: projected_output_resource_facts(
                structural_buckling_receipt_payload_bytes_upper_bound(),
            )?,
            output_set: projected_output_set_resource_facts(),
        },
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan != expected_resource_plan
        || resource_plan.input.phase != FeaResourcePhase::StructuralBuckling(workload.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "structural buckling resource plan does not converge from authoritative inputs",
        ));
    }
    let result = super::with_admitted_cpu_pool(&resource_plan, || {
        runmat_analysis_fea::execute_structural_buckling_workload(&workload, &mesh, &state)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
    })?;
    let receipt =
        StructuralBucklingNumericalReceipt::new(invocation.host.workload.identity, &result)?;
    Ok(vec![
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::SolverResult,
            receipt.source_result().digest(),
            result.canonical_bytes()?,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::NumericalReceipt,
            receipt.identity().digest(),
            receipt.canonical_bytes()?,
        ),
    ])
}
