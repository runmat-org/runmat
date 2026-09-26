use super::*;

pub(super) fn execute_structural_modal_extraction(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralModalWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let constraint_input = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let prestress_input =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    if invocation.inputs.len() != 4 + usize::from(prestress_input.is_some())
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural modal inputs or outputs differ from the workload contract".into(),
        ));
    }

    let workload = decode_prepared_structural_modal_workload(workload_input)?;
    let prestress_state = prestress_input
        .map(decode_prepared_semantic_state)
        .transpose()?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
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
        || constraint_input.manifest.logical_identity != expected_constraint_identity.digest()
        || prestress_input.is_some_and(|input| {
            prestress_state
                .as_ref()
                .is_none_or(|state| input.manifest.logical_identity != state.identity.digest())
        })
        || resource_input.manifest.logical_identity != resource_plan.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "structural modal inputs differ from their canonical identities",
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
            "structural modal task names a different authored procedure step",
        ));
    }

    let fields = structural_observable_field_definitions()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let observations =
        runmat_analysis_fea::results::observation::resolve_structural_observation_plan(
            workload.procedure_step()?,
            workload.model(),
            Some(&mesh),
            &constraints,
            &fields,
            ObservationLimits::default(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let result_payload_upper_bound =
        structural_modal_result_payload_bytes_upper_bound(Some(&mesh), &constraints, &workload)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        prestress_state.as_ref(),
        &resource_plan.request.accelerators,
        PreparedStructuralModalResourceFacts {
            inputs: prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            result_output: projected_output_resource_facts(result_payload_upper_bound)?,
            receipt_output: projected_output_resource_facts(
                structural_modal_receipt_payload_bytes_upper_bound(),
            )?,
            state_profiles_output:
                runmat_analysis_execution::structural_state_profiles_output_facts(
                    workload.model(),
                    workload.procedure_step()?,
                    &[SemanticStatePublicationSource::ModalBasis],
                    Digest::from_bytes(*mesh.canonical_digest.bytes()),
                )?,
            output_set: projected_output_set_resource_facts(),
        },
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan != expected_resource_plan
        || resource_plan.input.phase
            != FeaResourcePhase::StructuralModalExtraction(workload.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "structural modal resource plan does not converge from authoritative inputs",
        ));
    }

    let result = match prestress_state.as_ref() {
        Some(state) => runmat_analysis_fea::execute_structural_prestressed_modal_workload(
            &workload, &mesh, state,
        ),
        None => runmat_analysis_fea::execute_structural_modal_workload(&workload, &mesh),
    }
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let receipt = StructuralModalNumericalReceipt::new(invocation.host.workload.identity, &result)?;
    let payload = result.canonical_bytes()?;
    let profiles = runmat_analysis_execution::structural_state_publication_profiles(
        workload.model(),
        workload.procedure_step()?,
        receipt.source_result(),
        Digest::sha256(&payload),
        runmat_analysis_execution::StructuralStateProfileResult::Modal(&result),
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
use runmat_execution::Digest;
