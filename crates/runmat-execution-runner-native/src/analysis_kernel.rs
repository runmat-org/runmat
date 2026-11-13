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
    ResultPublicationRequest, SemanticStatePublicationSource, StructuralBucklingNumericalReceipt,
    StructuralFrequencySolverPlan, StructuralFrequencyStateInputs, StructuralFrequencyWorkload,
    StructuralLinearWorkload, StructuralLinearWorkloadResult, StructuralModalNumericalReceipt,
    StructuralModalResponseSpectrumNumericalReceipt, StructuralModalResponseSpectrumSolverResult,
    StructuralRandomResponseNumericalReceipt, StructuralRandomResponseSolverResult,
    StructuralTransientBlockRequest, StructuralTransientCheckpoint,
    StructuralTransientNumericalReceipt, StructuralTransientSolverResult,
    StructuralTransientWorkload,
};
#[cfg(test)]
use runmat_analysis_fea::execution::{
    SemanticStatePublicationRequest, StructuralLinearNumericalReceiptArtifact,
};
use runmat_analysis_fea::planning::resource::{
    build_observation_assembly_resource_plan, build_observation_history_checkpoint_resource_plan,
    build_observation_partition_resource_plan, build_observation_reduction_resource_plan,
    build_structural_buckling_resource_plan,
    build_structural_frequency_entry_resource_plan_from_compilation,
    build_structural_frequency_recovery_resource_plan_from_compilation,
    build_structural_frequency_result_assembly_resource_plan,
    build_structural_transient_block_resource_plan, observation_artifact_payload_bytes_upper_bound,
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
    structural_modal_result_output_resource_facts,
    structural_random_response_assembly_payload_bytes_upper_bound,
    structural_random_response_interval_payload_bytes_upper_bound,
    structural_random_response_receipt_payload_bytes_upper_bound, validate_resource_usage_evidence,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, ObservationPartitionResourcePlanning,
    ObservationPartitionSourceWorkspace, PreparedPhaseResourceFacts,
    PreparedStructuralBucklingResourceFacts, PreparedStructuralModalResourceFacts,
    ResourceUsageEvidence, StructuralFrequencyRecoveryResourceInput,
};
use runmat_analysis_fea::post::fields::{
    structural_observable_field_definitions, take_structural_observable_fields,
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
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

mod accuracy_evaluation;
mod acoustic_frequency_assembly;
mod acoustic_harmonic;
mod buckling;
#[cfg(test)]
mod buckling_tests;
mod acoustic_response;
mod electrical_static;
mod field_recovery;
mod fluid_steady;
mod fluid_transient;
mod frequency;
mod history_finalization;
mod magnetostatic;
mod maxwell_harmonic;
mod modal;
#[cfg(test)]
mod modal_tests;
mod nonlinear_block;
mod observation;
mod random_response;
#[cfg(test)]
mod random_response_tests;
mod response_spectrum;
#[cfg(test)]
mod response_spectrum_tests;
mod state_publication;
#[cfg(test)]
mod structural_adaptation_tests;
#[cfg(test)]
mod structural_fixture;
mod structural_linear;
mod thermal;
mod transient;
mod transient_scope;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAnalysisKernel;

impl AnalysisKernel for NativeAnalysisKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        match invocation.kind {
            AnalysisWorkKind::SolveNonlinearSolidBlock => nonlinear_block::execute(invocation),
            AnalysisWorkKind::Solve => structural_linear::execute(invocation),
            AnalysisWorkKind::SolveThermal => thermal::execute(invocation),
            AnalysisWorkKind::SolveAcousticHarmonic => acoustic_harmonic::execute(invocation),
            AnalysisWorkKind::AssembleAcousticFrequencyResult => {
                acoustic_frequency_assembly::execute(invocation)
            }
            AnalysisWorkKind::SolveElectricalStatic => electrical_static::execute(invocation),
            AnalysisWorkKind::SolveAcousticResponse => acoustic_response::execute(invocation),
            AnalysisWorkKind::SolveMagnetostatic => magnetostatic::execute(invocation),
            AnalysisWorkKind::SolveMaxwellHarmonic => maxwell_harmonic::execute(invocation),
            AnalysisWorkKind::SolveFluidSteady => fluid_steady::execute(invocation),
            AnalysisWorkKind::SolveFluidTransientBlock => fluid_transient::execute(invocation),
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
            AnalysisWorkKind::EvaluateAccuracy => accuracy_evaluation::execute(invocation),
            AnalysisWorkKind::PublishResult => execute_result_publication(invocation),
            AnalysisWorkKind::PublishState => state_publication::execute(invocation),
            AnalysisWorkKind::CheckpointObservationHistory => {
                observation::execute_observation_history_checkpoint(invocation)
            }
            AnalysisWorkKind::FinalizeObservationHistory => {
                history_finalization::execute(invocation)
            }
            AnalysisWorkKind::SolveStructuralTransientBlock => transient::execute_block(invocation),
            AnalysisWorkKind::PrepareStructuralTransientCheckpointScope => {
                transient_scope::execute_scope_preparation(invocation)
            }
            AnalysisWorkKind::AssembleStructuralTransientResult => {
                transient::execute_assembly(invocation)
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

/// The local execution plan a post-process phase was admitted with: its plan's CPU request,
/// the same authority `with_admitted_cpu_pool` sizes the pool from.
pub(crate) fn admitted_local_execution(
    resource_plan: &FeaResourcePlan,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::LocalExecutionPlan> {
    runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
        resource_plan.request.cpu_millicores,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
}

/// Runs one kernel's numerical work on the admitted CPU pool sized by its resource plan.
///
/// The declared plan's CPU request is the single thread authority: host planning derived it
/// from the admitted local execution plan and the kernel has already proven it converges
/// from authoritative inputs. Progress context and tracing span follow the work onto the pool.
fn with_admitted_cpu_pool<T: Send>(
    resource_plan: &FeaResourcePlan,
    work: impl FnOnce() -> AnalysisBridgeResult<T> + Send,
) -> AnalysisBridgeResult<T> {
    let threads = usize::from(
        runmat_analysis_fea::planning::resource::LocalThreadCount::from_cpu_millicores(
            resource_plan.request.cpu_millicores,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
        .get(),
    );
    let progress = runmat_analysis_fea::progress::capture_fea_progress_context();
    let span = tracing::Span::current();
    runmat_cpu_execution::with_cpu_pool(threads, || {
        let _progress = runmat_analysis_fea::replace_fea_progress_context(progress);
        span.in_scope(work)
    })
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
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
