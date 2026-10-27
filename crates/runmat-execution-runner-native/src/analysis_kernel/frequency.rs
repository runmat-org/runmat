use super::*;

pub(super) fn execute_structural_frequency_entry(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralFrequencyEntryRequest,
    )?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let states = semantic_state_inputs(&invocation)?;
    if invocation.inputs.len() != 4 + states.len()
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::StructuralFrequencyEntryResult]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency entry inputs or output differ from the workload contract".into(),
        ));
    }
    let workload = decode_prepared_structural_frequency_workload(workload_input)?;
    let request = decode_prepared_structural_frequency_entry_request(request_input)?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let prestress_states = states
        .iter()
        .filter(|state| state.kind == StateKind::GeometricStiffness)
        .collect::<Vec<_>>();
    let modal_states = states
        .iter()
        .filter(|state| state.kind == StateKind::ModalBasis)
        .collect::<Vec<_>>();
    if prestress_states.len() > 1
        || modal_states.len() > 1
        || prestress_states.len() + modal_states.len() != states.len()
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency entry has an invalid semantic-state inventory".into(),
        ));
    }
    let prestress_state = prestress_states.first().copied();
    let modal_state = modal_states.first().copied();
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let modal_state_matches = match (workload.solver(), modal_state) {
        (StructuralFrequencySolverPlan::Direct { .. }, None) => true,
        (StructuralFrequencySolverPlan::ModalReduced { basis_state }, Some(state)) => {
            state.identity == *basis_state
        }
        _ => false,
    };
    let prestress_state_matches = match (workload.prestress_transition(), prestress_state) {
        (None, None) => true,
        (Some(transition), Some(state)) => transition.source_state == state.identity,
        _ => false,
    };
    let state_topology_matches = states.iter().all(|state| {
        state.topology_digest
            == runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
    });
    if workload_input.manifest.logical_identity != workload.identity().digest()
        || mesh_input.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        || invocation.host.workload.study != request.study()
        || invocation.host.workload.step != request.step()
        || invocation.host.workload.execution_plan != request.execution_plan()
        || !modal_state_matches
        || !prestress_state_matches
        || !state_topology_matches
        || resource_plan_input.manifest.logical_identity != resource_plan.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency entry inputs differ from their exact identities",
        ));
    }
    let all_input_facts = prepared_input_resource_facts(invocation.inputs)?;
    let compiler_limit =
        runmat_analysis_fea::planning::resource::constraint_compilation_available_bytes(
            resource_plan.request.memory_bytes.min(
                workload
                    .procedure_step()?
                    .accuracy
                    .resources
                    .resident_byte_budget(),
            ),
            &all_input_facts,
            resource_plan.input.worker.fixed_memory_bytes,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let compiled = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        workload.model(),
        Some(&mesh),
        compiler_limit,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_structural_frequency_entry_resource_plan_from_compilation(
        &mesh,
        &compiled,
        &workload,
        &request,
        StructuralFrequencyStateInputs {
            prestress: prestress_state,
            modal_basis: modal_state,
        },
        &resource_plan.request.accelerators,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            output: projected_output_resource_facts(
                structural_frequency_entry_payload_bytes_upper_bound(&request)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            output_set: projected_output_set_resource_facts(),
        },
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
            resource_plan.request.cpu_millicores,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan != expected_resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "frequency entry resource plan does not converge from authoritative inputs",
        ));
    }
    drop(compiled);
    let result = super::with_admitted_cpu_pool(&resource_plan, || {
        Ok(match prestress_state {
            Some(state) => runmat_analysis_fea::pipeline::structural_frequency::execute_prestressed_structural_frequency_entry(
                &workload,
                &mesh,
                &request,
                state,
                modal_state,
            )?,
            None => runmat_analysis_fea::pipeline::structural_frequency::execute_structural_frequency_entry(
                &workload,
                &mesh,
                &request,
                modal_state,
            )?,
        })
    })?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::StructuralFrequencyEntryResult,
        result.identity().digest(),
        result.canonical_bytes()?,
    )])
}

pub(super) fn execute_structural_frequency_result_assembly(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralFrequencyResultAssemblyRequest,
    )?;
    let request = decode_prepared_structural_frequency_result_assembly_request(request_input)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let entry_inputs = invocation
        .inputs
        .iter()
        .filter(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::StructuralFrequencyEntryResult
        })
        .collect::<Vec<_>>();
    if invocation.inputs.len() != request.entries().len() + 2
        || entry_inputs.len() != request.entries().len()
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency assembly inputs or outputs differ from the workload contract"
                .into(),
        ));
    }
    let entries = entry_inputs
        .into_iter()
        .map(decode_prepared_structural_frequency_entry_result)
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let first = entries.first().ok_or_else(|| {
        AnalysisBridgeError::Invalid("structural frequency assembly has no entries".into())
    })?;
    if invocation.host.workload.study != first.request().study()
        || invocation.host.workload.step != first.request().step()
        || invocation.host.workload.execution_plan != first.request().execution_plan()
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency assembly has the wrong task authority",
        ));
    }
    request.validate_for(&entries)?;
    let FeaResourceDemand::StructuralFrequencyResultAssembly(_) = &resource_plan.input.demand
    else {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency assembly requires its exact phase resource demand",
        ));
    };
    let numerical_inputs = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .cloned()
        .collect::<Vec<_>>();
    let input_facts = prepared_input_resource_facts(&numerical_inputs)?;
    let output_record_count = u64::try_from(entries.len())
        .map_err(|_| AnalysisBridgeError::Invalid("frequency entry count overflows".into()))?
        .checked_add(1)
        .ok_or_else(|| AnalysisBridgeError::Invalid("frequency output count overflows".into()))?;
    let result_payload_upper_bound =
        structural_frequency_result_payload_bytes_upper_bound(&input_facts, output_record_count)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let receipt_payload_upper_bound =
        structural_frequency_receipt_payload_bytes_upper_bound(&input_facts, output_record_count)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_structural_frequency_result_assembly_resource_plan(
        &request,
        &entries,
        input_facts,
        projected_output_resource_facts(result_payload_upper_bound)?,
        projected_output_resource_facts(receipt_payload_upper_bound)?,
        projected_output_set_resource_facts(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan_input.manifest.logical_identity != resource_plan.identity.digest()
        || resource_plan.input.phase
            != FeaResourcePhase::StructuralFrequencyResultAssembly(request.identity())
        || !resource_plan.request.accelerators.is_empty()
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "frequency result resource plan does not converge from authoritative inputs",
        ));
    }
    let assembled = request.assemble(&entries)?;
    let mut partitions = Vec::with_capacity(assembled.result.samples().len() + 1);
    partitions.push(AnalysisKernelOutputPartition {
        logical_identity: assembled.result.manifest().identity().digest(),
        payload: assembled.result.manifest().canonical_bytes()?,
    });
    partitions.extend(
        assembled
            .result
            .samples()
            .iter()
            .map(|sample| {
                Ok(AnalysisKernelOutputPartition {
                    logical_identity: sample.identity().digest(),
                    payload: sample.canonical_bytes()?,
                })
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?,
    );
    Ok(vec![
        AnalysisKernelOutput::partitioned(
            AnalysisNumericalObjectKind::SolverResult,
            assembled.result.manifest().identity().digest(),
            partitions,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::NumericalReceipt,
            assembled.receipt.identity().digest(),
            assembled.receipt.canonical_bytes()?,
        ),
    ])
}
