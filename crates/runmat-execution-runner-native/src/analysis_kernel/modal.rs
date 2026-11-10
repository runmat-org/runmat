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

    let original_inputs = prepared_input_resource_facts(invocation.inputs)?;
    let compiler_limit =
        runmat_analysis_fea::planning::resource::constraint_compilation_available_bytes(
            resource_plan.request.memory_bytes.min(
                workload
                    .procedure_step()?
                    .accuracy
                    .resources
                    .resident_byte_budget(),
            ),
            &original_inputs,
            resource_plan.input.worker.fixed_memory_bytes,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let compiled = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        workload.model(),
        Some(&mesh),
        compiler_limit,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if compiled.plan() != &constraints {
        return Err(AnalysisBridgeError::Identity(
            "modal constraint plan differs from original bounded compilation",
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
    let result_output =
        structural_modal_result_output_resource_facts(Some(&mesh), &constraints, &workload)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    // The submitted plan records its solve path; the re-derivation reproduces that path and the
    // solver follows it.
    let path = runmat_analysis_fea::planning::resource::recorded_modal_solve_path(&resource_plan)
        .ok_or_else(|| {
        AnalysisBridgeError::Invalid("structural modal resource plan records no solve path".into())
    })?;
    let expected_resource_plan =
        runmat_analysis_fea::planning::resource::build_structural_modal_resource_plan_selecting(
        Some(&mesh),
        &compiled,
        &observations,
        &workload,
        prestress_state.as_ref(),
        &resource_plan.request.accelerators,
        PreparedStructuralModalResourceFacts {
            local_execution:
                runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                    resource_plan.request.cpu_millicores,
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            inputs: prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            result_output,
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
        runmat_analysis_fea::planning::resource::ModalSolveSelection::Recorded(path),
    )
    .map_err(AnalysisBridgeError::Resource)?;
    if resource_plan != expected_resource_plan
        || resource_plan.input.phase
            != FeaResourcePhase::StructuralModalExtraction(workload.identity())
    {
        return Err(AnalysisBridgeError::Identity(
            "structural modal resource plan does not converge from authoritative inputs",
        ));
    }

    drop(compiled);
    let cpu_threads = usize::from(
        runmat_analysis_fea::planning::resource::LocalThreadCount::from_cpu_millicores(
            resource_plan.request.cpu_millicores,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
        .get(),
    );
    let progress = runmat_analysis_fea::progress::capture_fea_progress_context();
    let span = tracing::Span::current();
    let result = runmat_cpu_execution::with_cpu_pool(cpu_threads, || {
        let _progress = runmat_analysis_fea::replace_fea_progress_context(progress);
        span.in_scope(|| match prestress_state.as_ref() {
            Some(state) => {
                runmat_analysis_fea::execute_structural_prestressed_modal_workload_on_path(
                    &workload, &mesh, state, path,
                )
            }
            None => runmat_analysis_fea::execute_structural_modal_workload_on_path(
                &workload, &mesh, path,
            ),
        })
    })
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
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
