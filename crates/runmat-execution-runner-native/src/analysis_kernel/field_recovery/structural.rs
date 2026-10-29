//! Publish the original static solve fields under freshly rederived admission.
use super::*;
use runmat_analysis_execution::decode_admitted_resource_plan;
pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let resource_objects = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let resource_plan = decode_admitted_resource_plan(
        resource_objects,
        invocation.host.output_policy.inventory_limits(),
    )?;
    runmat_analysis_execution::require_structural_recovery_decode_budget(
        invocation.inputs,
        &resource_plan,
        invocation.host.output_policy.inventory_limits(),
    )?;
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
    )?;
    let solver_invocation =
        unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let solver_result = unique_input(&invocation, AnalysisNumericalObjectKind::SolverResult)?;
    let mesh = optional_unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    if invocation.inputs.len() != 4 + usize::from(mesh.is_some())
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::Field]
    {
        return Err(AnalysisBridgeError::Invalid(
            "field recovery requires one request, source invocation, solver result, optional mesh, and field output".into(),
        ));
    }
    let request_manifest_identity = request.manifest.logical_identity;
    let request = FieldRecoveryRequest::from_canonical_bytes(&request.payload())?;
    if request.identity().digest() != request_manifest_identity
        || request.producer != invocation.host.workload.step
        || request.source.solver_result().map(|result| result.digest())
            != Some(solver_result.manifest.logical_identity)
    {
        return Err(AnalysisBridgeError::Identity(
            "field recovery inputs differ from the admitted source request",
        ));
    }
    let workload = StructuralLinearWorkload::from_canonical_bytes(&solver_invocation.payload())?;
    let mesh = mesh
        .map(|mesh| {
            SolverMeshArtifact::canonical_decode(&mesh.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    let topology = StructuralRecoveredFieldTopology::derive(&workload.model, mesh.as_ref())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if request.topology
        != (runmat_analysis_fea::execution::FieldTopologyIdentity::Structural {
            topology: topology.identity,
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "field recovery topology differs from its admitted identity",
        ));
    }
    let mut solver_result =
        StructuralLinearWorkloadResult::from_canonical_bytes(&solver_result.payload())?;
    solver_result.validate_thermal_workload(&workload)?;
    solver_result.validate_thermal_source(
        request.producer,
        request
            .source
            .solver_result()
            .ok_or(AnalysisBridgeError::Identity(
                "structural recovery requires a solver result source",
            ))?,
    )?;
    if solver_result.constraint_result.topology != topology.identity {
        return Err(AnalysisBridgeError::Identity(
            "field recovery topology differs from the completed solve",
        ));
    }
    let all_input_facts = prepared_input_resource_facts(invocation.inputs)?;
    let compiler_limit =
        runmat_analysis_fea::planning::resource::constraint_compilation_available_bytes(
            workload
                .procedure_step()?
                .accuracy
                .resources
                .resident_byte_budget()
                .min(resource_plan.request.memory_bytes),
            &all_input_facts,
            resource_plan.input.worker.fixed_memory_bytes,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let compilation = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        &workload.model,
        mesh.as_ref(),
        compiler_limit,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan = compilation.plan();
    solver_result
        .constraint_result
        .validate_against_model(constraint_plan, &workload.model, mesh.as_ref())
        .map_err(|_| {
            AnalysisBridgeError::Identity(
                "field recovery constraint result differs from the canonical executed plan",
            )
        })?;
    let expected_recovery_plan =
        runmat_analysis_fea::execution::FieldRecoveryPlan::resolve_structural(
            workload.procedure_step()?,
            topology,
            constraint_plan.groups.len(),
        )?;
    if request.plan != expected_recovery_plan {
        return Err(AnalysisBridgeError::Identity(
            "field recovery plan differs from the admitted workload and topology",
        ));
    }
    let expected_resource_plan =
        runmat_analysis_execution::build_structural_recovery_metadata_resource_plan(
            &request,
            &workload,
            mesh.as_ref(),
            &solver_result,
            &compilation,
            invocation.inputs,
            invocation.host.output_policy,
            crate::analysis_kernel::admitted_local_execution(&resource_plan)?,
        )?;
    if resource_plan != expected_resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "static field recovery resource plan differs from original construction",
        ));
    }
    drop(compilation);
    let mut fields = take_structural_observable_fields(&mut solver_result.run.fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    fields.extend(
        recover_constraint_group_observable_fields(&solver_result.constraint_result)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    );
    request
        .plan
        .validate_definitions(fields.iter().map(|field| field.definition().clone()))?;
    let mut entity_counts = topology.entity_counts;
    entity_counts.constraint_groups = solver_result.constraint_result.groups.len();
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        topology.identity.discretization_digest(),
        entity_counts,
        fields,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    recovered_field_output(&artifact)
}
