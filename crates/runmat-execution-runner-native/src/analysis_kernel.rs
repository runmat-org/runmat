use runmat_analysis_core::StateKind;
use runmat_analysis_execution::{
    build_prepared_structural_transient_result_resource_plan,
    decode_prepared_observation_history_checkpoint, decode_prepared_semantic_state,
    decode_prepared_structural_buckling_workload,
    decode_prepared_structural_frequency_entry_request,
    decode_prepared_structural_frequency_entry_result,
    decode_prepared_structural_frequency_result_assembly_request,
    decode_prepared_structural_frequency_workload,
    decode_prepared_structural_modal_response_spectrum_assembly_request,
    decode_prepared_structural_modal_response_spectrum_direction_request,
    decode_prepared_structural_modal_response_spectrum_direction_result,
    decode_prepared_structural_modal_response_spectrum_workload,
    decode_prepared_structural_modal_workload,
    decode_prepared_structural_random_response_assembly_request,
    decode_prepared_structural_random_response_interval_request,
    decode_prepared_structural_random_response_interval_result,
    decode_prepared_structural_random_response_workload,
    decode_prepared_structural_transient_result_assembly_request,
    decode_prepared_structural_transient_segment, prepared_input_resource_facts,
    prepared_input_resource_facts_without_resource_plan, projected_output_resource_facts,
    projected_output_set_resource_facts, validate_observation_resource_usage, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisKernel, AnalysisKernelInvocation, AnalysisKernelOutput,
    AnalysisKernelOutputPartition,
};
use runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan;
use runmat_analysis_fea::execution::{
    AccuracyEvaluationRequest, AnalysisNumericalObjectKind, AnalysisWorkKind,
    CanonicalSemanticState, ConstraintPlanIdentity, FieldRecoveryRequest, FieldRecoverySource,
    ObservationAssemblyRequest, ObservationEvaluationRequest, ObservationHistoryCheckpointRequest,
    ObservationPartitionEvaluationRequest, ObservationPartitionReductionRequest,
    ResultPublicationRequest, SemanticStatePublicationRequest, SemanticStatePublicationSource,
    StructuralBucklingNumericalReceipt, StructuralFrequencySolverPlan,
    StructuralFrequencyStateInputs, StructuralFrequencyWorkload,
    StructuralLinearNumericalReceiptArtifact, StructuralLinearWorkload,
    StructuralLinearWorkloadResult, StructuralModalNumericalReceipt,
    StructuralModalResponseSpectrumNumericalReceipt, StructuralModalResponseSpectrumSolverResult,
    StructuralModalSolverResult, StructuralRandomResponseNumericalReceipt,
    StructuralRandomResponseSolverResult, StructuralTransientBlockRequest,
    StructuralTransientCheckpoint, StructuralTransientNumericalReceipt,
    StructuralTransientSolverResult, StructuralTransientWorkload,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_assembly_resource_plan, build_observation_history_checkpoint_resource_plan,
    build_observation_partition_resource_plan, build_observation_reduction_resource_plan,
    build_structural_buckling_resource_plan, build_structural_frequency_entry_resource_plan,
    build_structural_frequency_recovery_resource_plan,
    build_structural_frequency_result_assembly_resource_plan, build_structural_modal_resource_plan,
    build_structural_resource_plan, build_structural_transient_block_resource_plan,
    observation_artifact_payload_bytes_upper_bound,
    observation_history_checkpoint_payload_bytes_upper_bound,
    observation_partition_payload_bytes_upper_bound,
    observation_reduction_payload_bytes_upper_bound,
    structural_buckling_receipt_payload_bytes_upper_bound,
    structural_buckling_result_payload_bytes_upper_bound,
    structural_frequency_entry_payload_bytes_upper_bound,
    structural_frequency_receipt_payload_bytes_upper_bound,
    structural_frequency_recovery_payload_bytes_upper_bound,
    structural_frequency_result_payload_bytes_upper_bound,
    structural_modal_receipt_payload_bytes_upper_bound,
    structural_modal_response_spectrum_assembly_payload_bytes_upper_bound,
    structural_modal_response_spectrum_direction_payload_bytes_upper_bound,
    structural_modal_response_spectrum_receipt_payload_bytes_upper_bound,
    structural_modal_result_payload_bytes_upper_bound,
    structural_random_response_assembly_payload_bytes_upper_bound,
    structural_random_response_interval_payload_bytes_upper_bound,
    structural_random_response_receipt_payload_bytes_upper_bound, validate_resource_usage_evidence,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, ObservationPartitionResourcePlanning,
    PreparedPhaseResourceFacts, PreparedStructuralBucklingResourceFacts,
    PreparedStructuralModalResourceFacts, ResourceUsageEvidence,
    StructuralFrequencyRecoveryResourceInput,
};
use runmat_analysis_fea::post::fields::{
    structural_observable_field_definitions, structural_static_observable_field_definitions,
    take_structural_observable_fields,
};
use runmat_analysis_fea::post::recover_constraint_group_observable_fields;
use runmat_analysis_fea::results::accuracy::AccuracyArtifact;
use runmat_analysis_fea::results::field::{
    canonical_field_partitions, RecoveredFieldArtifact, RecoveredFieldPartitionSet,
    StructuralRecoveredFieldTopology,
};
use runmat_analysis_fea::results::observation::{
    evaluate_structural_observation_partition,
    evaluate_structural_observation_partition_accelerated, reduce_observation_partitions,
    ObservationArtifact, ObservationArtifactInput, ObservationLimits,
    ObservationPartitionContribution, ObservationPartitionPlan, ObservationResult,
    StructuralObservationAccelerationInput,
};
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_analysis_fea::{
    run_linear_static_with_prepared_constraint_plan, LinearStaticSolveOptions,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

mod buckling;
#[cfg(test)]
mod buckling_tests;
mod field_recovery;
mod frequency;
mod history_finalization;
mod modal;
mod observation;
mod random_response;
#[cfg(test)]
mod random_response_tests;
mod response_spectrum;
#[cfg(test)]
mod response_spectrum_tests;
mod state_publication;
mod structural_linear;
mod thermal;
mod transient_scope;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAnalysisKernel;

impl AnalysisKernel for NativeAnalysisKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        match invocation.kind {
            AnalysisWorkKind::Solve => structural_linear::execute(invocation),
            AnalysisWorkKind::SolveThermal => thermal::execute(invocation),
            AnalysisWorkKind::RecoverField => field_recovery::execute_field_recovery(invocation),
            AnalysisWorkKind::EvaluateObservationPartition => {
                observation::execute_observation_partition(invocation)
            }
            AnalysisWorkKind::ReduceObservationPartitions => {
                observation::execute_observation_partition_reduction(invocation)
            }
            AnalysisWorkKind::AssembleObservations => {
                observation::execute_observation_assembly(invocation)
            }
            AnalysisWorkKind::EvaluateAccuracy => execute_accuracy_evaluation(invocation),
            AnalysisWorkKind::PublishResult => execute_result_publication(invocation),
            AnalysisWorkKind::PublishState => state_publication::execute(invocation),
            AnalysisWorkKind::CheckpointObservationHistory => {
                observation::execute_observation_history_checkpoint(invocation)
            }
            AnalysisWorkKind::FinalizeObservationHistory => {
                history_finalization::execute(invocation)
            }
            AnalysisWorkKind::SolveStructuralTransientBlock => {
                execute_structural_transient_block(invocation)
            }
            AnalysisWorkKind::PrepareStructuralTransientCheckpointScope => {
                transient_scope::execute_scope_preparation(invocation)
            }
            AnalysisWorkKind::AssembleStructuralTransientResult => {
                execute_structural_transient_result_assembly(invocation)
            }
            AnalysisWorkKind::SolveStructuralFrequencyEntry => {
                frequency::execute_structural_frequency_entry(invocation)
            }
            AnalysisWorkKind::ExtractStructuralModes => {
                modal::execute_structural_modal_extraction(invocation)
            }
            AnalysisWorkKind::SolveStructuralBucklingModes => {
                buckling::execute_structural_buckling(invocation)
            }
            AnalysisWorkKind::SolveStructuralModalResponseSpectrumDirection => {
                response_spectrum::execute_direction(invocation)
            }
            AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult => {
                response_spectrum::execute_assembly(invocation)
            }
            AnalysisWorkKind::SolveStructuralRandomResponseIntervals => {
                random_response::execute_intervals(invocation)
            }
            AnalysisWorkKind::AssembleStructuralRandomResponseResult => {
                random_response::execute_assembly(invocation)
            }
            AnalysisWorkKind::AssembleStructuralFrequencyResult => {
                frequency::execute_structural_frequency_result_assembly(invocation)
            }
            _ => Err(AnalysisBridgeError::Invalid(
                "native analysis kernel does not yet implement this work kind".into(),
            )),
        }
    }
}

fn execute_structural_transient_result_assembly(
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

fn execute_structural_transient_block(
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
    let segment = runmat_analysis_fea::pipeline::transient::run_transient_block(
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
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
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

fn execute_result_publication(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request = unique_input(&invocation, AnalysisNumericalObjectKind::PublicationRequest)?;
    let accuracy = unique_input(&invocation, AnalysisNumericalObjectKind::AccuracyReport)?;
    let resource_plan = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let resource_usage = unique_input(&invocation, AnalysisNumericalObjectKind::ResourceUsage)?;
    let observations =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::ObservationCatalog)?;
    if invocation.inputs.len() != 4 + usize::from(observations.is_some())
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::PublishedResult]
    {
        return Err(AnalysisBridgeError::Invalid(
            "result publication inputs or output differ from the workload contract".into(),
        ));
    }
    let request_manifest_identity = request.manifest.logical_identity;
    let request = ResultPublicationRequest::from_canonical_bytes(&request.payload())?;
    let accuracy_manifest_identity = accuracy.manifest.logical_identity;
    let accuracy = AccuracyArtifact::from_canonical_bytes(&accuracy.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan_manifest_identity = resource_plan.manifest.logical_identity;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_usage_manifest_identity = resource_usage.manifest.logical_identity;
    let resource_usage = ResourceUsageEvidence::from_canonical_bytes(&resource_usage.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let source_plan = resource_usage.primary_plan()?;
    validate_resource_usage_evidence(source_plan, &resource_usage)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if request.identity().digest() != request_manifest_identity
        || request.producer != invocation.host.workload.step
        || accuracy.identity().digest() != accuracy_manifest_identity
        || request.source_accuracy != accuracy.identity()
        || request.accuracy_request != accuracy.evaluation_request
        || request.source_result != accuracy.source_result
        || request.source_observations != accuracy.source_observations
        || request.source_observations.is_some() != observations.is_some()
        || resource_plan.identity.digest() != resource_plan_manifest_identity
        || resource_plan
            != runmat_analysis_execution::build_prepared_result_publication_resource_plan(
                &request,
                &accuracy,
                &resource_usage,
                invocation.inputs,
            )?
        || resource_usage.identity().digest() != resource_usage_manifest_identity
        || resource_usage.result_digest != request.source_result.digest()
        || !resource_usage.overruns.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "publication inputs differ from the admitted request",
        ));
    }
    let observations = observations
        .map(|objects| {
            let manifest_identity = objects.manifest.logical_identity;
            let artifact = runmat_analysis_fea::results::observation::ObservationPublicationCatalog::from_canonical_bytes(&objects.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if artifact.identity().digest() != manifest_identity
                || Some(artifact.identity()) != request.source_observations
            {
                return Err(AnalysisBridgeError::Identity(
                    "publication observations differ from the admitted request",
                ));
            }
            Ok(artifact)
        })
        .transpose()?;
    if let Some(observations) = &observations {
        validate_observation_resource_usage(observations, &resource_usage)?;
    }
    let expected_request =
        ResultPublicationRequest::new(&accuracy, observations.as_ref(), &resource_usage)?;
    if expected_request != request {
        return Err(AnalysisBridgeError::Identity(
            "publication request does not converge from its source artifacts",
        ));
    }
    let result = PublishedAnalysisResult::new(&request, &accuracy, &resource_usage)?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::PublishedResult,
        result.identity().digest(),
        result.canonical_bytes()?,
    )])
}

fn execute_accuracy_evaluation(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
    )?;
    let numerical_receipt =
        unique_input(&invocation, AnalysisNumericalObjectKind::NumericalReceipt)?;
    let observations =
        optional_unique_input(&invocation, AnalysisNumericalObjectKind::ObservationCatalog)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let source_context: Vec<_> = invocation
        .inputs
        .iter()
        .filter(|input| {
            matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::ThermalWorkload
                    | AnalysisNumericalObjectKind::Mesh
                    | AnalysisNumericalObjectKind::AccuracySourceContext
                    | AnalysisNumericalObjectKind::SolverInvocation
                    | AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload
                    | AnalysisNumericalObjectKind::StructuralRandomResponseWorkload
            )
        })
        .collect();
    if invocation.inputs.len() != 3 + usize::from(observations.is_some()) + source_context.len()
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::AccuracyReport]
    {
        return Err(AnalysisBridgeError::Invalid(
            "accuracy evaluation inputs or report output differ from the workload contract".into(),
        ));
    }
    let request_manifest_identity = request.manifest.logical_identity;
    let receipt_manifest_identity = numerical_receipt.manifest.logical_identity;
    let request = AccuracyEvaluationRequest::from_canonical_bytes(&request.payload())?;
    if request.identity().digest() != request_manifest_identity
        || request.producer != invocation.host.workload.step
        || request.source_observations.is_some() != observations.is_some()
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy inputs differ from the admitted evaluation request",
        ));
    }
    let observations = if let Some(observations) = observations {
        let manifest_identity = observations.manifest.logical_identity;
        let observations =
            runmat_analysis_fea::results::observation::ObservationPublicationCatalog::from_canonical_bytes(&observations.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        observations.validate_source_result(
            request.producer,
            request.source_kind,
            request.source_result,
        )?;
        if observations.identity().digest() != manifest_identity
            || Some(observations.identity()) != request.source_observations
        {
            return Err(AnalysisBridgeError::Identity(
                "accuracy observation artifact differs from its admitted source",
            ));
        }
        Some(observations)
    } else {
        None
    };
    let numerical_receipt =
        runmat_analysis_fea::execution::AccuracySourceReceipt::from_canonical_bytes(
            request.source_kind,
            &numerical_receipt.payload(),
        )?;
    if numerical_receipt.identity().digest() != receipt_manifest_identity
        || numerical_receipt.source_result() != request.source_result
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy numerical receipt differs from its admitted source result",
        ));
    }
    let admitted_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())?;
    let expected_plan = runmat_analysis_execution::build_prepared_accuracy_resource_plan(
        &request,
        &numerical_receipt,
        invocation.inputs,
    )?;
    if resource_plan_input.manifest.logical_identity != admitted_plan.identity.digest()
        || admitted_plan != expected_plan
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy resource plan differs from admitted inputs",
        ));
    }
    // Authenticate the exact admitted resource plan before reconstructing source topology
    // and constraints. The authored source checks remain mandatory after admission.
    runmat_analysis_execution::validate_accuracy_source_context(
        &request,
        &numerical_receipt,
        observations.as_ref(),
        &source_context,
    )?;
    let evidence = numerical_receipt.evidence(&request)?;
    let artifact =
        runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(&request, evidence)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::AccuracyReport,
        artifact.identity().digest(),
        artifact
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )])
}

fn constraint_plan_manifest_identity(
    invocation: &AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<runmat_execution::Digest> {
    Ok(
        unique_input(invocation, AnalysisNumericalObjectKind::ConstraintPlan)?
            .manifest
            .logical_identity,
    )
}

fn unique_input<'a>(
    invocation: &'a AnalysisKernelInvocation<'_>,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&'a runmat_analysis_execution::PreparedAnalysisObjects> {
    let mut matches = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == kind);
    let input = matches.next().ok_or_else(|| {
        AnalysisBridgeError::Invalid(format!("analysis workload is missing its {kind:?} input"))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}

fn inputs_except_resource_plan(
    invocation: &AnalysisKernelInvocation<'_>,
) -> Vec<runmat_analysis_execution::PreparedAnalysisObjects> {
    invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .cloned()
        .collect()
}

fn optional_unique_input<'a>(
    invocation: &'a AnalysisKernelInvocation<'_>,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<Option<&'a runmat_analysis_execution::PreparedAnalysisObjects>> {
    let mut matches = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == kind);
    let input = matches.next();
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}

fn semantic_state_inputs(
    invocation: &AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<CanonicalSemanticState>> {
    invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::SemanticState)
        .map(|input| {
            let state = decode_prepared_semantic_state(input)?;
            if input.manifest.logical_identity != state.identity.digest() {
                return Err(AnalysisBridgeError::Identity(
                    "semantic-state input differs from its object identity",
                ));
            }
            Ok(state)
        })
        .collect()
}

#[cfg(test)]
#[path = "analysis_kernel/observation_history_tests.rs"]
mod observation_history_tests;

#[cfg(test)]
#[path = "analysis_kernel/frequency_tests.rs"]
mod frequency_tests;

#[cfg(test)]
#[path = "analysis_kernel/transient_block_tests.rs"]
mod transient_block_tests;

#[cfg(test)]
#[path = "analysis_kernel/tests.rs"]
mod tests;
