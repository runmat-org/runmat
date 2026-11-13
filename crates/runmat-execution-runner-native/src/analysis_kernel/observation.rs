use super::*;
mod partition;
mod sampled;
pub(super) use partition::execute_observation_partition;

pub(super) fn execute_observation_history_checkpoint(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationHistoryCheckpointRequest,
    )?;
    let sample = unique_input(&invocation, AnalysisNumericalObjectKind::ObservationResult)?;
    let previous = optional_unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
    )?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    if invocation.inputs.len() != 3 + usize::from(previous.is_some())
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::ObservationHistoryCheckpoint]
    {
        return Err(AnalysisBridgeError::Invalid(
            "history checkpoint inputs or output differ from the workload contract".into(),
        ));
    }
    let request_identity = request.manifest.logical_identity;
    let sample_identity = sample.manifest.logical_identity;
    let request = ObservationHistoryCheckpointRequest::from_canonical_bytes(&request.payload())?;
    let sample = ObservationResult::from_canonical_bytes(&sample.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if request.identity.digest() != request_identity
        || sample.logical_identity.digest() != sample_identity
        || request.producer != invocation.host.workload.step
    {
        return Err(AnalysisBridgeError::Identity(
            "history checkpoint request or sample differs from its object identity",
        ));
    }
    let previous = previous
        .map(decode_prepared_observation_history_checkpoint)
        .transpose()?;
    request.validate_advance_inputs(previous.as_ref(), &sample)?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let FeaResourceDemand::StructuralObservationHistoryCheckpoint(_) = &resource_plan.input.demand
    else {
        return Err(AnalysisBridgeError::Identity(
            "history checkpoint requires its exact phase resource demand",
        ));
    };
    let input_facts = prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?;
    let output_payload_upper_bound = observation_history_checkpoint_payload_bytes_upper_bound(
        &request,
        previous.as_ref(),
        &sample,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_observation_history_checkpoint_resource_plan(
        &request,
        previous.as_ref(),
        &sample,
        input_facts,
        projected_output_resource_facts(output_payload_upper_bound)?,
        projected_output_set_resource_facts(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan_input.manifest.logical_identity != resource_plan.identity.digest()
        || resource_plan.input.phase
            != FeaResourcePhase::StructuralObservationHistoryCheckpoint(request.identity)
        || !resource_plan.request.accelerators.is_empty()
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "history checkpoint resource plan does not converge from authoritative inputs",
        ));
    }
    let checkpoint = request.advance(previous, sample)?;
    let mut partitions = Vec::with_capacity(checkpoint.samples().len() + 1);
    partitions.push(AnalysisKernelOutputPartition {
        logical_identity: checkpoint.manifest().identity.digest(),
        payload: checkpoint
            .manifest()
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    });
    partitions.extend(
        checkpoint
            .samples()
            .iter()
            .map(|sample| {
                Ok(AnalysisKernelOutputPartition {
                    logical_identity: sample.identity.digest(),
                    payload: sample
                        .canonical_bytes()
                        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                })
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?,
    );
    Ok(vec![AnalysisKernelOutput::partitioned(
        AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
        checkpoint.manifest().identity.digest(),
        partitions,
    )])
}

pub(super) fn execute_observation_partition_reduction(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let source_request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
    )?;
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationPartitionReductionRequest,
    )?;
    let partition_plan = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationPartitionPlan,
    )?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let contribution_inputs = invocation
        .inputs
        .iter()
        .filter(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::ObservationPartitionContribution
        })
        .collect::<Vec<_>>();
    if invocation.inputs.len() != 4 + contribution_inputs.len()
        || contribution_inputs.is_empty()
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::ObservationResult]
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation reduction inputs or result output differ from the workload contract"
                .into(),
        ));
    }
    let source_request_identity = source_request.manifest.logical_identity;
    let request_identity = request.manifest.logical_identity;
    let partition_plan_identity = partition_plan.manifest.logical_identity;
    let source_request =
        ObservationEvaluationRequest::from_canonical_bytes(&source_request.payload())?;
    let request = ObservationPartitionReductionRequest::from_canonical_bytes(&request.payload())?;
    let partition_plan = ObservationPartitionPlan::from_canonical_bytes(&partition_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let contributions = contribution_inputs
        .into_iter()
        .map(|input| {
            let contribution =
                ObservationPartitionContribution::from_canonical_bytes(&input.payload())
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if contribution.identity.digest() != input.manifest.logical_identity {
                return Err(AnalysisBridgeError::Identity(
                    "observation contribution differs from its object identity",
                ));
            }
            Ok(contribution)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    request.validate_for(&source_request, &partition_plan, &contributions)?;
    if source_request.identity().digest() != source_request_identity
        || request.identity().digest() != request_identity
        || partition_plan.identity.digest() != partition_plan_identity
        || resource_plan.identity.digest() != resource_plan_input.manifest.logical_identity
        || request.producer != invocation.host.workload.step
    {
        return Err(AnalysisBridgeError::Identity(
            "observation reduction inputs differ from their object identities",
        ));
    }
    let FeaResourceDemand::StructuralObservationReduction(_) = &resource_plan.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "observation reduction requires its exact phase resource demand",
        ));
    };
    let input_facts = prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?;
    let output_payload_upper_bound =
        observation_reduction_payload_bytes_upper_bound(&input_facts, &contributions)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_observation_reduction_resource_plan(
        &source_request,
        &request,
        &partition_plan,
        &contributions,
        input_facts,
        projected_output_resource_facts(output_payload_upper_bound)?,
        projected_output_set_resource_facts(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan.input.phase
        != FeaResourcePhase::StructuralObservationReduction(request.identity())
        || !resource_plan.request.accelerators.is_empty()
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "observation reduction resource plan does not converge from authoritative inputs",
        ));
    }
    let result =
        reduce_observation_partitions(&source_request.plan, &partition_plan, contributions)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::ObservationResult,
        result.logical_identity.digest(),
        result
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )])
}

pub(super) fn execute_observation_assembly(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let source_request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
    )?;
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationAssemblyRequest,
    )?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let result_inputs = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ObservationResult)
        .collect::<Vec<_>>();
    if invocation.inputs.len() != 3 + result_inputs.len()
        || result_inputs.is_empty()
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::Observation]
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation assembly inputs or artifact output differ from the workload contract"
                .into(),
        ));
    }
    let source_request_identity = source_request.manifest.logical_identity;
    let request_identity = request.manifest.logical_identity;
    let source_request =
        ObservationEvaluationRequest::from_canonical_bytes(&source_request.payload())?;
    let request = ObservationAssemblyRequest::from_canonical_bytes(&request.payload())?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let results = result_inputs
        .into_iter()
        .map(|input| {
            let result =
                runmat_analysis_fea::results::observation::ObservationResult::from_canonical_bytes(
                    &input.payload(),
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if result.logical_identity.digest() != input.manifest.logical_identity {
                return Err(AnalysisBridgeError::Identity(
                    "observation result differs from its object identity",
                ));
            }
            Ok(result)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    request.validate_for(&source_request, &results)?;
    if source_request.identity().digest() != source_request_identity
        || request.identity().digest() != request_identity
        || resource_plan.identity.digest() != resource_plan_input.manifest.logical_identity
        || request.producer != invocation.host.workload.step
    {
        return Err(AnalysisBridgeError::Identity(
            "observation assembly inputs differ from their object identities",
        ));
    }
    let FeaResourceDemand::StructuralObservationAssembly(_) = &resource_plan.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "observation assembly requires its exact phase resource demand",
        ));
    };
    let input_facts = prepared_input_resource_facts(&inputs_except_resource_plan(&invocation))?;
    let output_payload_upper_bound =
        observation_artifact_payload_bytes_upper_bound(&input_facts, results.len())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan = build_observation_assembly_resource_plan(
        &source_request,
        &request,
        &results,
        input_facts,
        projected_output_resource_facts(output_payload_upper_bound)?,
        projected_output_set_resource_facts(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan.input.phase
        != FeaResourcePhase::StructuralObservationAssembly(request.identity())
        || !resource_plan.request.accelerators.is_empty()
        || resource_plan != expected_resource_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "observation assembly resource plan does not converge from authoritative inputs",
        ));
    }
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: source_request.identity(),
        producer: source_request.producer,
        source: source_request.selection.source(),
        source_selection: source_request.selection.identity(),
        topology_digest: source_request.plan.topology_digest,
        plan_identity: source_request.plan.identity,
        results,
    })
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    observation_artifact_output(&artifact)
}

fn observation_artifact_output(
    artifact: &ObservationArtifact,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let manifest = artifact
        .manifest()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mut partitions = Vec::with_capacity(artifact.results().len() + 1);
    partitions.push(AnalysisKernelOutputPartition {
        logical_identity: artifact.identity().digest(),
        payload: manifest
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    });
    for result in artifact.results() {
        partitions.push(AnalysisKernelOutputPartition {
            logical_identity: result.logical_identity.digest(),
            payload: result
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        });
    }
    Ok(vec![AnalysisKernelOutput::partitioned(
        AnalysisNumericalObjectKind::Observation,
        artifact.identity().digest(),
        partitions,
    )])
}
