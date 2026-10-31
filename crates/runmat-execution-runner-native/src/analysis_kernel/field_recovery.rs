use super::*;
mod acoustic;
mod acoustic_response;
mod electrical;
mod fluid;
mod magnetic;
mod maxwell;
mod nonlinear;
mod statistical;
mod structural;

pub(super) fn execute_field_recovery(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    if invocation
        .inputs
        .iter()
        .any(|i| i.manifest.kind == AnalysisNumericalObjectKind::NonlinearSolidWorkload)
    {
        return nonlinear::execute(invocation);
    }
    if invocation
        .inputs
        .iter()
        .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::FluidSteadyWorkload)
    {
        return fluid::execute(invocation);
    }
    if invocation
        .inputs
        .iter()
        .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::MaxwellHarmonicWorkload)
    {
        return maxwell::execute(invocation);
    }
    if invocation
        .inputs
        .iter()
        .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::AcousticHarmonicWorkload)
    {
        return acoustic::execute(invocation);
    }
    if invocation
        .inputs
        .iter()
        .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::AcousticResponseWorkload)
    {
        return acoustic_response::execute(invocation);
    }
    if invocation
        .inputs
        .iter()
        .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::MagnetostaticWorkload)
    {
        return magnetic::execute(invocation);
    }
    if invocation
        .inputs
        .iter()
        .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::ElectricalStaticWorkload)
    {
        return electrical::execute(invocation);
    }
    if statistical::is_statistical(&invocation) {
        return statistical::execute(invocation);
    }
    if invocation.inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::StructuralFrequencyWorkload
    }) {
        return execute_structural_frequency_field_recovery(invocation);
    }
    structural::execute(invocation)
}

pub(super) fn recovered_field_output(
    artifact: &RecoveredFieldArtifact,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let manifest = artifact
        .manifest()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    // Each field's storage partitions are materialized and encoded once, independently per
    // field (on the admitted pool when installed), and published in canonical field order.
    let encoded = runmat_cpu_execution::ordered_map(artifact.fields(), |field| {
        canonical_field_partitions(field)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
            .into_iter()
            .map(|partition| {
                Ok(AnalysisKernelOutputPartition {
                    logical_identity: partition.identity.digest(),
                    payload: partition
                        .canonical_bytes()
                        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                })
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()
    });
    let mut partitions = Vec::with_capacity(
        1 + encoded
            .iter()
            .map(|field| field.as_ref().map_or(0, Vec::len))
            .sum::<usize>(),
    );
    partitions.push(AnalysisKernelOutputPartition {
        logical_identity: artifact.identity().digest(),
        payload: manifest
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    });
    for field in encoded {
        partitions.extend(field?);
    }
    Ok(vec![AnalysisKernelOutput::partitioned(
        AnalysisNumericalObjectKind::Field,
        artifact.identity().digest(),
        partitions,
    )])
}

fn execute_structural_frequency_field_recovery(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
    )?;
    let workload = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralFrequencyWorkload,
    )?;
    let mesh = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let result = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralFrequencyEntryResult,
    )?;
    let resource_plan = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let states = semantic_state_inputs(&invocation)?;
    if invocation.inputs.len() != 5 + states.len()
        || states.len() > 1
        || states
            .iter()
            .any(|state| state.kind != StateKind::GeometricStiffness)
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::Field]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency field recovery requires one request, workload, mesh, entry result, and field output"
                .into(),
        ));
    }
    let request_identity = request.manifest.logical_identity;
    let request = FieldRecoveryRequest::from_canonical_bytes(&request.payload())?;
    let workload_identity = workload.manifest.logical_identity;
    let workload = decode_prepared_structural_frequency_workload(workload)?;
    let mesh_identity = mesh.manifest.logical_identity;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let result = decode_prepared_structural_frequency_entry_result(result)?;
    let prestress_state = states.first();
    let resource_plan_identity = resource_plan.manifest.logical_identity;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if request.identity().digest() != request_identity
        || request.producer != invocation.host.workload.step
        || request.source != FieldRecoverySource::StructuralFrequencyEntry(result.identity())
        || workload.identity().digest() != workload_identity
        || runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()) != mesh_identity
        || match (workload.prestress_transition(), prestress_state) {
            (None, None) => false,
            (Some(transition), Some(state)) => {
                transition.source_state != state.identity
                    || state.topology_digest
                        != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
            }
            _ => true,
        }
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency field recovery inputs differ from their object identities",
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
    let constraints = compiled.plan();
    let expected_resource_plan =
        build_structural_frequency_recovery_resource_plan_from_compilation(
            &mesh,
            &compiled,
            &workload,
            StructuralFrequencyRecoveryResourceInput {
                request: &request,
                result: &result,
            },
            StructuralFrequencyStateInputs {
                prestress: prestress_state,
                modal_basis: None,
            },
            &resource_plan.request.accelerators,
            PreparedPhaseResourceFacts {
                inputs: prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
                output: projected_output_resource_facts(
                    structural_frequency_recovery_payload_bytes_upper_bound(
                        &mesh,
                        constraints,
                        &workload,
                    )
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
    if resource_plan.identity.digest() != resource_plan_identity
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency recovery received a substituted resource plan",
        ));
    }
    drop(compiled);
    let artifact = super::with_admitted_cpu_pool(&resource_plan, || {
        Ok(match prestress_state {
            Some(state) => runmat_analysis_fea::pipeline::structural_frequency::recover_prestressed_structural_frequency_entry_fields(
                &workload,
                &mesh,
                &request,
                &result,
                state,
            )?,
            None => runmat_analysis_fea::pipeline::structural_frequency::recover_structural_frequency_entry_fields(
                &workload,
                &mesh,
                &request,
                &result,
            )?,
        })
    })?;
    recovered_field_output(&artifact)
}
