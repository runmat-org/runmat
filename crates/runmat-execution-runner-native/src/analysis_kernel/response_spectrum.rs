use super::*;

pub(super) fn execute_direction(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let state_input = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    let constraint_input = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionRequest,
    )?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len() != 6
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult]
    {
        return Err(AnalysisBridgeError::Invalid(
            "response-spectrum direction inputs or output differ from the workload contract".into(),
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
    let request =
        decode_prepared_structural_modal_response_spectrum_direction_request(request_input)?;
    if request_input.manifest.logical_identity != request.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum direction request has another logical identity",
        ));
    }
    request.validate_for(&decoded.workload)?;
    let prepared = super::with_admitted_cpu_pool(&decoded.resource_plan, || {
        Ok(
            runmat_analysis_fea::prepare_structural_modal_response_spectrum(
                &decoded.workload,
                &decoded.mesh,
                &decoded.state,
                &decoded.constraints,
                &decoded.resource_plan.request.accelerators,
            )?,
        )
    })?;
    if request.scope() != prepared.scope() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum direction request differs from the worker-reconstructed scope",
        ));
    }
    let expected_plan = prepared
        .resources()
        .plan_for_direction(
            &request,
            prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            projected_output_resource_facts(
                structural_modal_response_spectrum_direction_payload_bytes_upper_bound(&request)
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
            != FeaResourcePhase::StructuralModalResponseSpectrumDirection(request.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum direction resource plan does not converge from authoritative inputs",
        ));
    }
    let result = super::with_admitted_cpu_pool(&decoded.resource_plan, || {
        Ok(
            runmat_analysis_fea::execute_structural_modal_response_spectrum_direction(
                &decoded.workload,
                &prepared,
                &request,
            )?,
        )
    })?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult,
        result.identity().digest(),
        result.canonical_bytes()?,
    )])
}

pub(super) fn execute_assembly(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let state_input = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    let constraint_input = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralModalResponseSpectrumAssemblyRequest,
    )?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let result_inputs = invocation
        .inputs
        .iter()
        .filter(|input| {
            input.manifest.kind
                == AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult
        })
        .collect::<Vec<_>>();
    let request =
        decode_prepared_structural_modal_response_spectrum_assembly_request(request_input)?;
    if invocation.inputs.len() != 6 + request.entries().len()
        || result_inputs.len() != request.entries().len()
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "response-spectrum assembly inputs or outputs differ from the workload contract".into(),
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
            "response-spectrum assembly request has another logical identity",
        ));
    }
    let mut transported = result_inputs
        .into_iter()
        .map(|input| {
            let result =
                decode_prepared_structural_modal_response_spectrum_direction_result(input)?;
            if input.manifest.logical_identity != result.identity().digest() {
                return Err(AnalysisBridgeError::Identity(
                    "response-spectrum direction result has another logical identity",
                ));
            }
            Ok(result)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    transported.sort_by_key(|result| result.request().ordinal());
    if transported.as_slice() != request.entries() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum assembly result inputs differ from its canonical request",
        ));
    }
    let prepared = super::with_admitted_cpu_pool(&decoded.resource_plan, || {
        Ok(
            runmat_analysis_fea::prepare_structural_modal_response_spectrum(
                &decoded.workload,
                &decoded.mesh,
                &decoded.state,
                &decoded.constraints,
                &decoded.resource_plan.request.accelerators,
            )?,
        )
    })?;
    if request.scope() != prepared.scope() {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum assembly differs from the worker-reconstructed scope",
        ));
    }
    let expected_plan = prepared
        .resources()
        .plan_for_assembly(
            &request,
            prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            runmat_analysis_fea::planning::resource::PreparedStructuralResponseSpectrumAssemblyOutputs {
            result: projected_output_resource_facts(
                structural_modal_response_spectrum_assembly_payload_bytes_upper_bound(&request)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            receipt: projected_output_resource_facts(
                structural_modal_response_spectrum_receipt_payload_bytes_upper_bound(),
            )?,
            field_recovery: projected_output_resource_facts(
                prepared
                    .resources()
                    .field_recovery_payload_bytes_upper_bound()
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            state_profiles: runmat_analysis_execution::structural_state_profiles_output_facts(
                decoded.workload.model(),
                decoded.workload.procedure_step()?,
                &[SemanticStatePublicationSource::StructuralResponseEnvelope],
                request.scope().topology_digest(),
            )?,
            },
            projected_output_set_resource_facts(),
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                decoded.resource_plan.request.cpu_millicores,
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if decoded.resource_plan != expected_plan
        || decoded.resource_plan.input.phase
            != FeaResourcePhase::StructuralModalResponseSpectrumAssembly(request.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum assembly resource plan does not converge from authoritative inputs",
        ));
    }
    let result = StructuralModalResponseSpectrumSolverResult::new(&decoded.workload, request)?;
    let receipt = StructuralModalResponseSpectrumNumericalReceipt::new(
        invocation.host.workload.identity,
        &result,
    )?;
    let payload = result.canonical_bytes()?;
    let profiles = runmat_analysis_execution::structural_state_publication_profiles(
        decoded.workload.model(),
        decoded.workload.procedure_step()?,
        receipt.source_result(),
        Digest::sha256(&payload),
        runmat_analysis_execution::StructuralStateProfileResult::ResponseSpectrum(&result),
    )?;
    let field_recovery = super::with_admitted_cpu_pool(&decoded.resource_plan, || {
        Ok(
            runmat_analysis_fea::recover_structural_modal_response_spectrum_fields(
                &decoded.workload,
                &prepared,
                result.numerical(),
            )?,
        )
    })?;
    field_recovery
        .validate_for_result(result.numerical())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
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
            AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
            field_recovery
                .identity()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            field_recovery
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::StatePublicationProfiles,
            profiles.identity(),
            profiles.canonical_bytes()?,
        ),
    ])
}

struct DecodedPhysicalInputs {
    workload: runmat_analysis_fea::execution::StructuralModalResponseSpectrumWorkload,
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
    let workload = decode_prepared_structural_modal_response_spectrum_workload(workload_input)?;
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
            "response-spectrum physical inputs differ from their canonical identities",
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
