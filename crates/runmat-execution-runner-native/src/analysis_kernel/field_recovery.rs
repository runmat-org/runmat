use super::*;

pub(super) fn execute_field_recovery(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    if invocation.inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::StructuralFrequencyWorkload
    }) {
        return execute_structural_frequency_field_recovery(invocation);
    }
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
    )?;
    let solver_invocation =
        unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let solver_result = unique_input(&invocation, AnalysisNumericalObjectKind::SolverResult)?;
    let mesh = optional_unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    if invocation.inputs.len() != 3 + usize::from(mesh.is_some())
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
    if topology.identity != request.topology {
        return Err(AnalysisBridgeError::Identity(
            "field recovery topology differs from its admitted identity",
        ));
    }
    let mut solver_result =
        StructuralLinearWorkloadResult::from_canonical_bytes(&solver_result.payload())?;
    if solver_result.constraint_result.topology != topology.identity {
        return Err(AnalysisBridgeError::Identity(
            "field recovery topology differs from the completed solve",
        ));
    }
    let constraint_plan =
        CanonicalStructuralConstraintPlan::compile_for_model(&workload.model, mesh.as_ref())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    solver_result
        .constraint_result
        .validate_against_model(&constraint_plan, &workload.model, mesh.as_ref())
        .map_err(|_| {
            AnalysisBridgeError::Identity(
                "field recovery constraint result differs from the canonical executed plan",
            )
        })?;
    let mut fields = take_structural_observable_fields(&mut solver_result.run.fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    fields.extend(
        recover_constraint_group_observable_fields(&solver_result.constraint_result)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    );
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

fn recovered_field_output(
    artifact: &RecoveredFieldArtifact,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let manifest = artifact
        .manifest()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let storage_partition_count = artifact
        .fields()
        .iter()
        .try_fold(0usize, |count, field| {
            canonical_field_partitions(field)
                .map(|partitions| count.saturating_add(partitions.len()))
        })
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mut partitions = Vec::with_capacity(storage_partition_count + 1);
    partitions.push(AnalysisKernelOutputPartition {
        logical_identity: artifact.identity().digest(),
        payload: manifest
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    });
    for field in artifact.fields() {
        for partition in canonical_field_partitions(field)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
        {
            partitions.push(AnalysisKernelOutputPartition {
                logical_identity: partition.identity.digest(),
                payload: partition
                    .canonical_bytes()
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            });
        }
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
    if invocation.inputs.len() != 5
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
    let resource_plan_identity = resource_plan.manifest.logical_identity;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if request.identity().digest() != request_identity
        || request.producer != invocation.host.workload.step
        || request.source != FieldRecoverySource::StructuralFrequencyEntry(result.identity())
        || workload.identity().digest() != workload_identity
        || runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()) != mesh_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency field recovery inputs differ from their object identities",
        ));
    }
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(workload.model(), Some(&mesh))
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_structural_frequency_recovery_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        &result,
        &resource_plan.request.accelerators,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?,
            output: projected_output_resource_facts(
                structural_frequency_recovery_payload_bytes_upper_bound(
                    &mesh,
                    &constraints,
                    &workload,
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?,
            output_set: projected_output_set_resource_facts(),
        },
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan.identity.digest() != resource_plan_identity
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency recovery received a substituted resource plan",
        ));
    }
    let artifact =
        runmat_analysis_fea::pipeline::structural_frequency::recover_structural_frequency_entry_fields(
            &workload,
            &mesh,
            &request,
            &result,
        )?;
    recovered_field_output(&artifact)
}
