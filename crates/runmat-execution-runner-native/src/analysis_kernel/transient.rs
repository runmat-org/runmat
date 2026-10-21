//! Native typed transient blocks and result assembly, preserving the shared analysis ABI.
use super::*;

pub(super) fn execute_assembly(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralTransientResultAssemblyRequest,
    )?;
    let request = decode_prepared_structural_transient_result_assembly_request(request_input)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let segment_inputs = invocation
        .inputs
        .iter()
        .filter(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::StructuralTransientSegment
        })
        .collect::<Vec<_>>();
    if invocation.inputs.len() != request.segments.len() + 2
        || segment_inputs.len() != request.segments.len()
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
            ]
        || invocation.host.workload.study != request.scope.study
        || invocation.host.workload.step != request.scope.step
        || invocation.host.workload.execution_plan != request.scope.execution_plan
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient result inputs or outputs differ from the workload contract"
                .into(),
        ));
    }
    let numerical_inputs = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .cloned()
        .collect::<Vec<_>>();
    let input_facts = prepared_input_resource_facts(&numerical_inputs)?;
    let FeaResourceDemand::StructuralTransientResultAssembly(demand) = &resource_plan.input.demand
    else {
        return Err(AnalysisBridgeError::Identity(
            "transient result assembly requires its exact phase resource demand",
        ));
    };
    let segment_count = u64::try_from(request.segments.len())
        .map_err(|_| AnalysisBridgeError::Invalid("transient segment count overflows".into()))?;
    let sample_count = u64::try_from(request.scope.schedule.points.len())
        .map_err(|_| AnalysisBridgeError::Invalid("transient sample count overflows".into()))?;
    let physical_dof_count = u64::try_from(request.scope.dof_count).map_err(|_| {
        AnalysisBridgeError::Invalid("transient physical DOF count overflows".into())
    })?;
    if resource_plan_input.manifest.logical_identity != resource_plan.identity.digest()
        || resource_plan.input.phase
            != FeaResourcePhase::StructuralTransientResultAssembly(request.identity)
        || demand.scope_digest
            != request
                .scope
                .identity()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
                .digest()
        || demand.inputs != input_facts
        || demand.segment_count.0 != segment_count
        || demand.sample_count.0 != sample_count
        || demand.physical_dof_count.0 != physical_dof_count
    {
        return Err(AnalysisBridgeError::Identity(
            "transient result resource plan input facts differ from the prepared workload",
        ));
    }
    let decoded = segment_inputs
        .into_iter()
        .map(|input| {
            let segment = decode_prepared_structural_transient_segment(input)?;
            if input.manifest.logical_identity != segment.manifest().identity.digest() {
                return Err(AnalysisBridgeError::Identity(
                    "structural transient segment object has the wrong logical identity",
                ));
            }
            Ok(segment)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let mut segments = Vec::with_capacity(decoded.len());
    let mut remaining = decoded;
    for expected in &request.segments {
        let Some(position) = remaining
            .iter()
            .position(|segment| segment.manifest().identity == *expected)
        else {
            return Err(AnalysisBridgeError::Identity(
                "structural transient result assembly is missing a requested segment",
            ));
        };
        segments.push(remaining.remove(position));
    }
    if !remaining.is_empty() {
        return Err(AnalysisBridgeError::Identity(
            "structural transient result assembly contains an unrelated segment",
        ));
    }
    let expected_resource_plan = build_prepared_structural_transient_result_resource_plan(
        &request,
        &segments,
        &numerical_inputs,
    )?;
    if resource_plan_input.manifest.logical_identity != resource_plan.identity.digest()
        || resource_plan.input.phase
            != FeaResourcePhase::StructuralTransientResultAssembly(request.identity)
        || !resource_plan.request.accelerators.is_empty()
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "transient result resource plan does not converge from authoritative inputs",
        ));
    }
    let result = StructuralTransientSolverResult::assemble(&request, &segments)?;
    let receipt = StructuralTransientNumericalReceipt::new(&result, &request, &segments)?;
    let mut result_partitions = Vec::with_capacity(result.samples().len() + 1);
    result_partitions.push(AnalysisKernelOutputPartition {
        logical_identity: result.manifest().identity.digest(),
        payload: result.manifest().canonical_bytes()?,
    });
    result_partitions.extend(
        result
            .samples()
            .iter()
            .map(|sample| {
                Ok(AnalysisKernelOutputPartition {
                    logical_identity: sample.identity.digest(),
                    payload: sample.canonical_bytes()?,
                })
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?,
    );
    Ok(vec![
        AnalysisKernelOutput::partitioned(
            AnalysisNumericalObjectKind::SolverResult,
            result.manifest().identity.digest(),
            result_partitions,
        ),
        AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::NumericalReceipt,
            receipt.identity.digest(),
            receipt.canonical_bytes()?,
        ),
    ])
}

pub(super) fn execute_block(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let workload_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralTransientWorkload,
    )?;
    let mesh_input = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let request_input = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralTransientBlockRequest,
    )?;
    let predecessor_input = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StructuralTransientCheckpoint,
    )?;
    let modal_input =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
    let resource_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len()
        != 4 + usize::from(predecessor_input.is_some()) + usize::from(modal_input.is_some())
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::StructuralTransientSegment]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient block inputs or output differ from the workload contract".into(),
        ));
    }
    let workload = StructuralTransientWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let request = StructuralTransientBlockRequest::from_canonical_bytes(&request_input.payload())?;
    let predecessor = predecessor_input
        .map(|input| StructuralTransientCheckpoint::from_canonical_bytes(&input.payload()))
        .transpose()?;
    let modal_state = modal_input
        .map(decode_prepared_semantic_state)
        .transpose()?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if workload_input.manifest.logical_identity != request.scope.source_workload.digest()
        || mesh_input.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        || request_input.manifest.logical_identity != request.identity.digest()
        || predecessor_input
            .zip(predecessor.as_ref())
            .is_some_and(|(input, checkpoint)| {
                input.manifest.logical_identity != checkpoint.identity.digest()
            })
        || invocation.host.workload.study != request.scope.study
        || invocation.host.workload.step != request.scope.step
        || invocation.host.workload.execution_plan != request.scope.execution_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient block inputs differ from their exact scope or object identities",
        ));
    }
    let modal_state_matches = match (
        &workload.solver,
        &request.scope.coordinate_space,
        modal_state.as_ref(),
    ) {
        (
            runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
            runmat_analysis_fea::execution::StructuralTransientCoordinateSpace::Direct,
            None,
        ) => true,
        (
            runmat_analysis_fea::execution::StructuralTransientSolverPlan::ModalReduced {
                basis_transition,
            },
            runmat_analysis_fea::execution::StructuralTransientCoordinateSpace::ModalReduced {
                basis_state: scoped_basis,
                ..
            },
            Some(state),
        ) => {
            state.identity == basis_transition.source_state
                && state.identity == *scoped_basis
                && state.topology_digest
                    == runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        }
        _ => false,
    };
    if !modal_state_matches {
        return Err(AnalysisBridgeError::Identity(
            "structural transient modal state differs from its workload and checkpoint scope",
        ));
    }
    request.validate_against_predecessor(predecessor.as_ref())?;
    let expected_plan = build_structural_transient_block_resource_plan(
        &workload,
        &mesh,
        modal_state.as_ref(),
        &request,
        predecessor.as_ref(),
        Some(prepared_input_resource_facts_without_resource_plan(
            invocation.inputs,
        )?),
        Some(projected_output_set_resource_facts()),
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
            resource_plan.request.cpu_millicores,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_input.manifest.logical_identity != resource_plan.identity.digest()
        || resource_plan != expected_plan
        || resource_plan.input.phase != FeaResourcePhase::StructuralTransientBlock(request.identity)
        || !resource_plan.request.accelerators.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "transient block resource plan does not converge from authoritative inputs",
        ));
    }
    let segment = super::with_admitted_cpu_pool(&resource_plan, || {
        runmat_analysis_fea::pipeline::transient::run_transient_block(
            runmat_analysis_fea::pipeline::transient::StructuralTransientRunBlockInput {
                workload: &workload,
                solver_mesh: &mesh,
                request: &request,
                predecessor: predecessor.as_ref(),
                modal_state: modal_state.as_ref(),
                observation_cursors: predecessor
                    .as_ref()
                    .map(|checkpoint| checkpoint.observation_cursors.clone())
                    .unwrap_or_default(),
            },
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
    })?;
    let mut partitions = Vec::with_capacity(
        segment.samples().len() + 1 + usize::from(segment.initial_sample().is_some()),
    );
    partitions.push(AnalysisKernelOutputPartition {
        logical_identity: segment.manifest().identity.digest(),
        payload: segment.manifest().canonical_bytes()?,
    });
    if let Some(initial) = segment.initial_sample() {
        partitions.push(AnalysisKernelOutputPartition {
            logical_identity: initial.identity.digest(),
            payload: initial.canonical_bytes()?,
        });
    }
    partitions.extend(
        segment
            .samples()
            .iter()
            .map(|sample| {
                Ok(AnalysisKernelOutputPartition {
                    logical_identity: sample.identity.digest(),
                    payload: sample.canonical_bytes()?,
                })
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?,
    );
    Ok(vec![AnalysisKernelOutput::partitioned(
        AnalysisNumericalObjectKind::StructuralTransientSegment,
        segment.manifest().identity.digest(),
        partitions,
    )])
}
