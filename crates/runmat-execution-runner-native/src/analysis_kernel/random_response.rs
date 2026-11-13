use super::*;

pub(super) fn execute_intervals(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralRandomResponseWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let state_input = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    let constraint_input = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralRandomResponseIntervalRequest,
    )?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len() != 6
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult]
    {
        return Err(AnalysisBridgeError::Invalid(
            "random-response interval inputs or output differ from the workload contract".into(),
        ));
    }
    let decoded = decode_physical_inputs(
        &invocation,
        workload_input,
        mesh_input,
        state_input,
        constraint_input,
        resource_input,
    )?;
    let request = decode_prepared_structural_random_response_interval_request(request_input)?;
    if request_input.manifest.logical_identity != request.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "random-response interval request has another logical identity",
        ));
    }
    request.validate_for(&decoded.workload)?;
    let prepared = super::with_admitted_cpu_pool(&decoded.resource_plan, || {
        Ok(runmat_analysis_fea::prepare_structural_random_response(
            &decoded.workload,
            &decoded.mesh,
            &decoded.state,
            &decoded.constraints,
        )?)
    })?;
    if request.scope() != prepared.scope() {
        return Err(AnalysisBridgeError::Identity(
            "random-response interval request differs from the worker-reconstructed scope",
        ));
    }
    let expected_plan = prepared
        .resources()
        .plan_for_intervals(
            &request,
            prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            projected_output_resource_facts(
                structural_random_response_interval_payload_bytes_upper_bound(&request)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            projected_output_set_resource_facts(),
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                decoded.resource_plan.request.cpu_millicores,
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if decoded.resource_plan != expected_plan
        || decoded.resource_plan.input.phase
            != FeaResourcePhase::StructuralRandomResponseInterval(request.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "random-response interval resource plan does not converge from authoritative inputs",
        ));
    }
    let result = super::with_admitted_cpu_pool(&decoded.resource_plan, || {
        Ok(
            runmat_analysis_fea::execute_structural_random_response_intervals(
                &decoded.workload,
                &prepared,
                &request,
            )?,
        )
    })?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult,
        result.identity().digest(),
        result.canonical_bytes()?,
    )])
}

pub(super) fn execute_assembly(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralRandomResponseWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let state_input = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    let constraint_input = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralRandomResponseAssemblyRequest,
    )?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let task_inputs = invocation
        .inputs
        .iter()
        .filter(|input| {
            input.manifest.kind
                == AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult
        })
        .collect::<Vec<_>>();
    let request = decode_prepared_structural_random_response_assembly_request(request_input)?;
    if invocation.inputs.len() != 6 + request.tasks().len()
        || task_inputs.len() != request.tasks().len()
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "random-response assembly inputs or outputs differ from the workload contract".into(),
        ));
    }
    let decoded = decode_physical_inputs(
        &invocation,
        workload_input,
        mesh_input,
        state_input,
        constraint_input,
        resource_input,
    )?;
    request.scope().validate_for(&decoded.workload)?;
    if request_input.manifest.logical_identity != request.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "random-response assembly request has another logical identity",
        ));
    }
    let mut transported = task_inputs
        .into_iter()
        .map(|input| {
            let result = decode_prepared_structural_random_response_interval_result(input)?;
            if input.manifest.logical_identity != result.identity().digest() {
                return Err(AnalysisBridgeError::Identity(
                    "random-response interval result has another logical identity",
                ));
            }
            Ok(result)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    transported.sort_by_key(|result| result.request().task_ordinal());
    if transported.as_slice() != request.tasks() {
        return Err(AnalysisBridgeError::Identity(
            "random-response assembly result inputs differ from its canonical request",
        ));
    }
    let prepared = super::with_admitted_cpu_pool(&decoded.resource_plan, || {
        Ok(runmat_analysis_fea::prepare_structural_random_response(
            &decoded.workload,
            &decoded.mesh,
            &decoded.state,
            &decoded.constraints,
        )?)
    })?;
    if request.scope() != prepared.scope() {
        return Err(AnalysisBridgeError::Identity(
            "random-response assembly differs from the worker-reconstructed scope",
        ));
    }
    let expected_plan = prepared
        .resources()
        .plan_for_assembly(
            &request,
            prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            projected_output_resource_facts(
                structural_random_response_assembly_payload_bytes_upper_bound(&request)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            projected_output_resource_facts(
                structural_random_response_receipt_payload_bytes_upper_bound(),
            )?,
            runmat_analysis_execution::structural_state_profiles_output_facts(
                decoded.workload.model(),
                decoded.workload.procedure_step()?,
                &[SemanticStatePublicationSource::StructuralRandomStatistics],
                request.scope().topology_digest(),
            )?,
            projected_output_set_resource_facts(),
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                decoded.resource_plan.request.cpu_millicores,
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if decoded.resource_plan != expected_plan
        || decoded.resource_plan.input.phase
            != FeaResourcePhase::StructuralRandomResponseAssembly(request.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "random-response assembly resource plan does not converge from authoritative inputs",
        ));
    }
    let result = StructuralRandomResponseSolverResult::new(&decoded.workload, request)?;
    let receipt =
        StructuralRandomResponseNumericalReceipt::new(invocation.host.workload.identity, &result)?;
    let payload = result.canonical_bytes()?;
    let profiles = runmat_analysis_execution::structural_state_publication_profiles(
        decoded.workload.model(),
        decoded.workload.procedure_step()?,
        receipt.source_result(),
        Digest::sha256(&payload),
        runmat_analysis_execution::StructuralStateProfileResult::RandomResponse(&result),
    )?;
    Ok(vec![
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::SolverResult,
            receipt.source_result().digest(),
            payload,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::NumericalReceipt,
            receipt.identity().digest(),
            receipt.canonical_bytes()?,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::StatePublicationProfiles,
            profiles.identity(),
            profiles.canonical_bytes()?,
        ),
    ])
}

struct DecodedPhysicalInputs {
    workload: runmat_analysis_fea::execution::StructuralRandomResponseWorkload,
    mesh: SolverMeshArtifact,
    state: CanonicalSemanticState,
    constraints: CanonicalStructuralConstraintPlan,
    resource_plan: FeaResourcePlan,
}

fn decode_physical_inputs(
    invocation: &AnalysisKernelInvocation<'_>,
    workload_input: &runmat_analysis_execution::PreparedAnalysisObjects,
    mesh_input: &runmat_analysis_execution::PreparedAnalysisObjects,
    state_input: &runmat_analysis_execution::PreparedAnalysisObjects,
    constraint_input: &runmat_analysis_execution::PreparedAnalysisObjects,
    resource_input: &runmat_analysis_execution::PreparedAnalysisObjects,
) -> AnalysisBridgeResult<DecodedPhysicalInputs> {
    let workload = decode_prepared_structural_random_response_workload(workload_input)?;
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
    let constraint_identity = ConstraintPlanIdentity::new(
        invocation.host.workload.step,
        topology_digest,
        constraints
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    let expected_step = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
            &workload.model().sequence,
        )?,
        workload.procedure_step()?,
    )?;
    if workload_input.manifest.logical_identity != workload.identity().digest()
        || mesh_input.manifest.logical_identity != topology_digest
        || state_input.manifest.logical_identity != state.identity.digest()
        || constraint_input.manifest.logical_identity != constraint_identity.digest()
        || resource_input.manifest.logical_identity != resource_plan.identity.digest()
        || invocation.host.workload.step != expected_step
    {
        return Err(AnalysisBridgeError::Identity(
            "random-response physical inputs differ from their canonical identities",
        ));
    }
    Ok(DecodedPhysicalInputs {
        workload,
        mesh,
        state,
        constraints,
        resource_plan,
    })
}
use runmat_execution::Digest;
