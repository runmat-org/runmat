use runmat_analysis_core::{AccuracyTarget, ErrorContributionKind, RequiredAccuracyEvidence};
use runmat_analysis_execution::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernel, AnalysisKernelInvocation,
    AnalysisKernelOutput, AnalysisKernelOutputPartition,
};
use runmat_analysis_fea::constraints::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintResult,
};
use runmat_analysis_fea::execution::{
    AccuracyEvaluationRequest, AnalysisNumericalObjectKind, AnalysisWorkKind,
    CanonicalSemanticState, ConstraintPlanIdentity, FieldRecoveryRequest,
    ObservationAssemblyRequest, ObservationEvaluationRequest,
    ObservationPartitionEvaluationRequest, ObservationPartitionReductionRequest,
    ResultPublicationRequest, SemanticStatePublicationRequest,
    StructuralLinearNumericalReceiptArtifact, StructuralLinearWorkload,
    StructuralLinearWorkloadResult,
};
use runmat_analysis_fea::planning::resource::{
    build_structural_resource_plan, validate_resource_usage_evidence, FeaResourcePlan,
    ResourceUsageEvidence,
};
use runmat_analysis_fea::post::fields::{
    structural_observable_field_definitions, take_structural_observable_fields,
};
use runmat_analysis_fea::results::accuracy::{
    evaluate_accuracy, publication_gate, AccuracyArtifact, AccuracyArtifactInput, AccuracyEvidence,
    AccuracyEvidenceScope, AccuracyMeasure,
};
use runmat_analysis_fea::results::field::{
    canonical_field_partitions, RecoveredFieldArtifact, RecoveredFieldPartitionSet,
    StructuralRecoveredFieldTopology,
};
use runmat_analysis_fea::results::observation::{
    evaluate_structural_observation_partition, reduce_observation_partitions, ObservationArtifact,
    ObservationArtifactInput, ObservationArtifactManifest, ObservationLimits,
    ObservationPartitionContribution, ObservationPartitionPlan,
};
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_analysis_fea::{
    run_linear_static_with_prepared_constraint_plan, LinearStaticSolveOptions,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAnalysisKernel;

impl AnalysisKernel for NativeAnalysisKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        match invocation.kind {
            AnalysisWorkKind::Solve => execute_structural_linear(invocation),
            AnalysisWorkKind::RecoverField => execute_field_recovery(invocation),
            AnalysisWorkKind::EvaluateObservationPartition => {
                execute_observation_partition(invocation)
            }
            AnalysisWorkKind::ReduceObservationPartitions => {
                execute_observation_partition_reduction(invocation)
            }
            AnalysisWorkKind::AssembleObservations => execute_observation_assembly(invocation),
            AnalysisWorkKind::EvaluateAccuracy => execute_accuracy_evaluation(invocation),
            AnalysisWorkKind::PublishResult => execute_result_publication(invocation),
            AnalysisWorkKind::PublishState => execute_state_publication(invocation),
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
    validate_resource_usage_evidence(&resource_plan, &resource_usage)
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
            let artifact = ObservationArtifactManifest::from_canonical_bytes(&objects.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if artifact.identity.digest() != manifest_identity
                || Some(artifact.identity) != request.source_observations
            {
                return Err(AnalysisBridgeError::Identity(
                    "publication observations differ from the admitted request",
                ));
            }
            Ok(artifact)
        })
        .transpose()?;
    let expected_request = ResultPublicationRequest::new(&accuracy, observations.as_ref())?;
    if expected_request != request {
        return Err(AnalysisBridgeError::Identity(
            "publication request does not converge from its source artifacts",
        ));
    }
    let result = PublishedAnalysisResult::new(&request, &accuracy)?;
    Ok(vec![AnalysisKernelOutput {
        kind: AnalysisNumericalObjectKind::PublishedResult,
        logical_identity: result.identity().digest(),
        payload: result.canonical_bytes()?,
        partitions: Vec::new(),
    }])
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
    if invocation.inputs.len() != 2 + usize::from(observations.is_some())
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
    if let Some(observations) = observations {
        let manifest_identity = observations.manifest.logical_identity;
        let observations =
            ObservationArtifactManifest::from_canonical_bytes(&observations.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if observations.identity.digest() != manifest_identity
            || Some(observations.identity) != request.source_observations
            || Some(observations.evaluation_request) != request.observation_request
            || observations.producer != request.producer
            || observations.source_result != request.source_result
        {
            return Err(AnalysisBridgeError::Identity(
                "accuracy observation artifact differs from its admitted source",
            ));
        }
    }
    let numerical_receipt = StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
        &numerical_receipt.payload(),
    )?;
    if numerical_receipt.identity().digest() != receipt_manifest_identity
        || numerical_receipt.source_result != request.source_result
    {
        return Err(AnalysisBridgeError::Identity(
            "accuracy numerical receipt differs from its admitted source result",
        ));
    }
    let evidence = if request.plan.criteria.iter().any(|criterion| {
        criterion.target == AccuracyTarget::GlobalEquilibrium
            && criterion.contribution == ErrorContributionKind::Algebraic
            && criterion.evidence == RequiredAccuracyEvidence::ResidualBound
    }) {
        vec![AccuracyEvidence::new(
            AccuracyTarget::GlobalEquilibrium,
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
            AccuracyMeasure::Relative(
                numerical_receipt.receipt().relative_equilibrium_residual,
            ),
            vec![
                "canonical unconstrained equilibrium residual projected into the admissible constraint space"
                    .into(),
            ],
            AccuracyEvidenceScope::SolverResult(request.source_result),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?]
    } else {
        Vec::new()
    };
    let evaluation = evaluate_accuracy(&request.plan, &evidence)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let gate = publication_gate(&request.plan, request.formulation_requirement, &evaluation)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let artifact = AccuracyArtifact::new(AccuracyArtifactInput {
        evaluation_request: request.identity(),
        producer: request.producer,
        source_result: request.source_result,
        source_observations: request.source_observations,
        formulation_requirement: request.formulation_requirement,
        evidence,
        evaluation,
        gate,
    })
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok(vec![AnalysisKernelOutput {
        kind: AnalysisNumericalObjectKind::AccuracyReport,
        logical_identity: artifact.identity().digest(),
        payload: artifact
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        partitions: Vec::new(),
    }])
}

fn execute_observation_partition(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let source_request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
    )?;
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationPartitionEvaluationRequest,
    )?;
    let partition_plan = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationPartitionPlan,
    )?;
    let fields = unique_input(&invocation, AnalysisNumericalObjectKind::FieldPartition)?;
    let solver_invocation =
        unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let mesh = optional_unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    if invocation.inputs.len() != 6 + usize::from(mesh.is_some())
        || invocation.host.workload.outputs
            != vec![AnalysisNumericalObjectKind::ObservationPartitionContribution]
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation partition inputs or contribution output differ from the workload contract"
                .into(),
        ));
    }
    let source_request_identity = source_request.manifest.logical_identity;
    let request_identity = request.manifest.logical_identity;
    let partition_plan_identity = partition_plan.manifest.logical_identity;
    let field_set_identity = fields.manifest.logical_identity;
    let source_request =
        ObservationEvaluationRequest::from_canonical_bytes(&source_request.payload())?;
    let request = ObservationPartitionEvaluationRequest::from_canonical_bytes(&request.payload())?;
    let partition_plan = ObservationPartitionPlan::from_canonical_bytes(&partition_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let fields = RecoveredFieldPartitionSet::from_canonical_bytes(&fields.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let partition = partition_plan
        .partitions
        .iter()
        .find(|partition| partition.identity == request.partition)
        .ok_or(AnalysisBridgeError::Identity(
            "observation partition request names no canonical assignment",
        ))?;
    request.validate_for(
        &source_request,
        &partition_plan,
        partition,
        &fields,
        &resource_plan,
    )?;
    if source_request.identity().digest() != source_request_identity
        || request.identity().digest() != request_identity
        || partition_plan.identity.digest() != partition_plan_identity
        || fields.identity.digest() != field_set_identity
        || resource_plan.identity.digest() != resource_plan_input.manifest.logical_identity
        || request.producer != invocation.host.workload.step
    {
        return Err(AnalysisBridgeError::Identity(
            "observation partition inputs differ from their object identities",
        ));
    }
    let workload = StructuralLinearWorkload::from_canonical_bytes(&solver_invocation.payload())?;
    let mesh = mesh
        .map(|mesh| {
            SolverMeshArtifact::canonical_decode(&mesh.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    let contribution = evaluate_structural_observation_partition(
        &source_request.plan,
        &partition_plan,
        partition,
        &workload.model,
        mesh.as_ref(),
        &fields,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::ObservationPartitionContribution,
        contribution.identity.digest(),
        contribution
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )])
}

fn execute_observation_partition_reduction(
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
    request.validate_for(
        &source_request,
        &partition_plan,
        &contributions,
        &resource_plan,
    )?;
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

fn execute_observation_assembly(
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
    request.validate_for(&source_request, &results, &resource_plan)?;
    if source_request.identity().digest() != source_request_identity
        || request.identity().digest() != request_identity
        || resource_plan.identity.digest() != resource_plan_input.manifest.logical_identity
        || request.producer != invocation.host.workload.step
    {
        return Err(AnalysisBridgeError::Identity(
            "observation assembly inputs differ from their object identities",
        ));
    }
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: source_request.identity(),
        producer: source_request.producer,
        field_recovery_request: source_request.field_recovery_request,
        source_fields: source_request.source_fields,
        source_result: source_request.source_result,
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

fn execute_field_recovery(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
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
        || request.source_result.digest() != solver_result.manifest.logical_identity
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
    let fields = take_structural_observable_fields(&mut solver_result.run.fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        request.source_result,
        topology.identity.discretization_digest(),
        topology.entity_counts,
        fields,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
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

fn execute_structural_linear(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let solve_invocation =
        unique_input(&invocation, AnalysisNumericalObjectKind::SolverInvocation)?;
    let mesh_input = optional_unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let constraint_plan = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let solve = StructuralLinearWorkload::from_canonical_bytes(&solve_invocation.payload())?;
    let expected_inputs = 3
        + usize::from(mesh_input.is_some())
        + usize::from(solve.initial_guess_transition.is_some());
    if invocation.inputs.len() != expected_inputs
        || invocation.host.workload.outputs
            != vec![
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
            ]
    {
        return Err(AnalysisBridgeError::Invalid(
            "structural linear solve inputs differ from its invocation and solver-result contract"
                .into(),
        ));
    }
    let solver_model = solve.solver_model()?;
    let mesh = mesh_input
        .map(|input| {
            SolverMeshArtifact::canonical_decode(&input.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    if let (Some(input), Some(mesh)) = (mesh_input, mesh.as_ref()) {
        if runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
            != input.manifest.logical_identity
        {
            return Err(AnalysisBridgeError::Identity(
                "structural mesh content differs from its input identity",
            ));
        }
    }
    let constraint_plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&constraint_plan.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    match mesh.as_ref() {
        Some(mesh) => constraint_plan.validate_against(&solver_model, mesh),
        None => constraint_plan.validate_against_explicit(&solver_model),
    }
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let topology_digest = constraint_plan.topology.discretization_digest();
    let initial_displacement = solve
        .initial_guess_transition
        .as_ref()
        .map(|transition| {
            if transition.consumer != invocation.host.workload.step {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess transition targets a different solver step",
                ));
            }
            let state = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
            if state.manifest.logical_identity != transition.source_state.digest() {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess state manifest differs from its transition",
                ));
            }
            let state = CanonicalSemanticState::from_canonical_bytes(&state.payload())?;
            if state.identity != transition.source_state
                || state.state_id != transition.state_id
                || state.kind != runmat_analysis_core::StateKind::StructuralPrimary
                || state.topology_digest != topology_digest
                || transition.target_topology_digest != topology_digest
            {
                return Err(AnalysisBridgeError::Identity(
                    "initial-guess state differs from its transition or solver topology",
                ));
            }
            let field = state
                .fields
                .iter()
                .find(|field| {
                    field.field_id == runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
                })
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid(
                        "structural-primary state has no displacement field".into(),
                    )
                })?;
            field.as_host_f64().map(<[f64]>::to_vec).ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "structural-primary displacement is not host-readable".into(),
                )
            })
        })
        .transpose()?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan_input.manifest.logical_identity != resource_plan.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "resource-plan manifest differs from its canonical identity",
        ));
    }
    let field_definitions = structural_observable_field_definitions()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let observations =
        runmat_analysis_fea::results::observation::resolve_structural_observation_plan(
            solve.procedure_step()?,
            &solve.model,
            mesh.as_ref(),
            &field_definitions,
            ObservationLimits::default(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan =
        build_structural_resource_plan(mesh.as_ref(), &constraint_plan, &observations, &solve)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if expected_resource_plan != resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "resource plan does not converge from authoritative structural inputs",
        ));
    }
    let expected_constraint_identity = ConstraintPlanIdentity::new(
        invocation.host.workload.step,
        topology_digest,
        constraint_plan
            .content_digest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )?;
    if constraint_plan_manifest_identity(&invocation)? != expected_constraint_identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "constraint-plan manifest differs from the compiled plan",
        ));
    }
    let run = run_linear_static_with_prepared_constraint_plan(
        &solver_model,
        solve.backend,
        LinearStaticSolveOptions {
            formulation_requirement: solve.formulation_requirement,
            preconditioner_kind: solve.preconditioner,
            algebra_backend_kind: solve.algebra_backend,
            solver_mesh_artifact_path: None,
            solver_mesh: mesh.clone(),
            require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
            initial_displacement,
        },
        &constraint_plan,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_result = CanonicalStructuralConstraintResult::recover_for_model(
        &constraint_plan,
        &solver_model,
        mesh.as_ref(),
        &run.fields,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let result = StructuralLinearWorkloadResult::new(run, constraint_result)?;
    let payload = result.canonical_bytes()?;
    let numerical_receipt =
        StructuralLinearNumericalReceiptArtifact::new(invocation.host.workload.identity, &result)?;
    let result_identity = numerical_receipt.source_result;
    Ok(vec![
        AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::SolverResult,
            logical_identity: result_identity.digest(),
            payload,
            partitions: Vec::new(),
        },
        AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::NumericalReceipt,
            logical_identity: numerical_receipt.identity().digest(),
            payload: numerical_receipt.canonical_bytes()?,
            partitions: Vec::new(),
        },
    ])
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

fn execute_state_publication(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let publication = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StatePublicationRequest,
    )?;
    let solver_result = unique_input(&invocation, AnalysisNumericalObjectKind::SolverResult)?;
    if invocation.inputs.len() != 2
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::SemanticState]
    {
        return Err(AnalysisBridgeError::Invalid(
            "state publication requires one publication request, one solver result, and one semantic-state output"
                .into(),
        ));
    }
    let publication =
        SemanticStatePublicationRequest::from_canonical_bytes(&publication.payload())?;
    if solver_result.manifest.logical_identity != publication.source_result.digest() {
        return Err(AnalysisBridgeError::Identity(
            "state publication received a substituted solver result",
        ));
    }
    let solver_result =
        StructuralLinearWorkloadResult::from_canonical_bytes(&solver_result.payload())?;
    let state = CanonicalSemanticState::from_publication(&publication, &solver_result.run.fields)?;
    Ok(vec![AnalysisKernelOutput {
        kind: AnalysisNumericalObjectKind::SemanticState,
        logical_identity: state.identity.digest(),
        payload: state.canonical_bytes()?,
        partitions: Vec::new(),
    }])
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_analysis_execution::{
        decode_recovered_field_artifact, prepare_structural_linear_program,
        prepare_structural_sequence_step, AnalysisCapabilityRequirement, AnalysisHostWorkload,
        AnalysisWorkloadInput, AnalysisWorkloadRequest, StructuralLinearProgramPreparation,
    };
    use runmat_analysis_fea::constraints::CanonicalStructuralTopologyIdentity;
    use runmat_analysis_fea::execution::SolverResultIdentity;
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::planning::resource::{
        record_resource_usage, AnalysisResourceUsage, ByteCount,
    };
    use runmat_analysis_fea::results::accuracy::{AccuracyStatus, PublicationDecision};
    use runmat_analysis_fea::results::field::{
        RecoveredField, RecoveredFieldCatalog, RecoveredFieldSelection,
    };
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::ComputeBackend;
    use runmat_execution::resource::ResourceUsage;
    use runmat_execution::value::ValueRefKind;
    use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::object::ObjectInventoryLimits;
    use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    use super::*;

    fn digest(label: &str) -> Digest {
        Digest::sha256(label)
    }

    fn revision() -> ProgramRevision {
        ProgramRevision::new(
            digest("graph"),
            digest("source"),
            ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
        )
        .unwrap()
    }

    fn prepare_kernel_output(
        output: &AnalysisKernelOutput,
    ) -> runmat_analysis_execution::PreparedAnalysisObjects {
        if output.partitions.is_empty() {
            return runmat_analysis_execution::prepare_analysis_objects(
                output.kind,
                output.logical_identity,
                &output.payload,
                4096,
                ObjectNamespace::ResultValue,
                ObjectInventoryLimits::default(),
            )
            .unwrap();
        }
        let partitions = output
            .partitions
            .iter()
            .map(
                |partition| runmat_analysis_fea::execution::AnalysisPayloadPartition {
                    logical_identity: partition.logical_identity,
                    payload: &partition.payload,
                },
            )
            .collect::<Vec<_>>();
        runmat_analysis_execution::prepare_partitioned_analysis_objects(
            output.kind,
            output.logical_identity,
            &partitions,
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap()
    }

    fn test_host_for_inputs(
        source: &AnalysisHostWorkload,
        kind: AnalysisWorkKind,
        inputs: &[runmat_analysis_execution::PreparedAnalysisObjects],
        outputs: Vec<AnalysisNumericalObjectKind>,
        result_input: Option<usize>,
    ) -> AnalysisHostWorkload {
        let workload_inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if result_input == Some(index) {
                    ValueRefKind::ResultObject
                } else {
                    ValueRefKind::DriverObject
                },
            })
            .collect();
        AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                source.workload.study,
                source.workload.step,
                source.workload.execution_plan,
                kind,
                workload_inputs,
                outputs,
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            source.artifact_access.clone(),
        )
        .unwrap()
    }

    #[test]
    fn native_kernel_executes_path_free_structural_linear_workload() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut producer_step = runmat_analysis_core::ProcedureStep::structural_static("preload");
        producer_step.observations = vec![structural_displacement_observation()];
        let state_id = producer_step.state_outputs[0].state_id.clone();
        let mut consumer_step = runmat_analysis_core::ProcedureStep::structural_static("branch");
        consumer_step
            .state_inputs
            .push(runmat_analysis_core::StateInput {
                input_id: "warm-start".into(),
                required_kind: runmat_analysis_core::StateKind::StructuralPrimary,
                source: runmat_analysis_core::StateSource::Upstream {
                    producer_step: producer_step.step_id.clone(),
                    state_id: state_id.clone(),
                },
                use_as: runmat_analysis_core::StateUse::InitialGuess,
                conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
            });
        let authored_transition = runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId("warm-start".into()),
            producer_step: producer_step.step_id.clone(),
            consumer_step: consumer_step.step_id.clone(),
            state_id,
            method: runmat_analysis_core::StateTransitionMethod::InitialGuessOnly,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        };
        model.sequence = runmat_analysis_core::AnalysisSequence {
            sequence_id: "stateful-sequence".into(),
            steps: vec![producer_step, consumer_step],
            dependencies: vec![runmat_analysis_core::SequenceDependency {
                predecessor: authored_transition.producer_step.clone(),
                successor: authored_transition.consumer_step.clone(),
                reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
            }],
            transitions: vec![authored_transition.clone()],
            branches: Vec::new(),
        };
        let step_id = model.sequence.steps[0].step_id.clone();
        let solve = StructuralLinearWorkload {
            schema_version: 1,
            model,
            step_id,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                    std::num::NonZeroU32::new(2).unwrap(),
                ),
            require_solver_mesh_for_solid: false,
            initial_guess_transition: None,
        };
        let mut mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        for (face_index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
            face.provenance.push(PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: if face_index == 0 { "root" } else { "tip" }.into(),
                assembly_path: vec!["root".into()],
            });
            face.provenance.sort();
        }
        if let Some(assignment) = solve.model.material_assignments.first() {
            let region = PersistentEntityId {
                kind: PersistentEntityKind::Region,
                source_topology_id: assignment.region_id.clone(),
                assembly_path: vec!["root".into()],
            };
            mesh.topology.volume_elements[0].region_id = region.clone();
            mesh.topology.regions[0].region_id = region;
        }
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();
        let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
            authored_study_digest: digest("study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "study:test".into(),
                encryption_context: digest("encryption"),
            },
            workload: solve.clone(),
            mesh: Some(mesh),
            initial_guess: None,
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        })
        .unwrap();
        let host = prepared.host;
        let inputs = prepared.inputs;
        let study = host.workload.study;
        let step = host.workload.step;
        let execution_plan = host.workload.execution_plan;
        let outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &host,
                inputs: &inputs,
            })
            .unwrap();
        let result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&outputs[0].payload).unwrap();
        assert!(!result.run.fields_are_empty());
        let executed_constraint_plan =
            CanonicalStructuralConstraintPlan::from_canonical_bytes(&inputs[2].payload()).unwrap();
        result
            .constraint_result
            .validate_against(&executed_constraint_plan)
            .unwrap();
        assert!(result
            .run
            .diagnostics
            .iter()
            .all(|diagnostic| !diagnostic.message.contains("artifact_path")));

        let mut substituted_plan = inputs.clone();
        substituted_plan[2].manifest.logical_identity = digest("substituted-constraint-plan");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &host,
                inputs: &substituted_plan,
            })
            .is_err());

        let solver_result = prepare_kernel_output(&outputs[0]);
        let numerical_receipt = prepare_kernel_output(&outputs[1]);
        let field_source_result = SolverResultIdentity::new(
            host.workload.identity,
            solver_result.manifest.payload_digest,
        )
        .unwrap();
        assert_eq!(
            StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
                &numerical_receipt.payload()
            )
            .unwrap()
            .source_result,
            field_source_result
        );
        let field_recovery_request =
            FieldRecoveryRequest::new(step, field_source_result, executed_constraint_plan.topology)
                .unwrap();
        let field_recovery_request = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::FieldRecoveryRequest,
            field_recovery_request.identity().digest(),
            &field_recovery_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let field_inputs = vec![
            field_recovery_request,
            inputs[0].clone(),
            solver_result.clone(),
            inputs[1].clone(),
        ];
        let field_workload_inputs = field_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if index == 2 {
                    ValueRefKind::ResultObject
                } else {
                    ValueRefKind::DriverObject
                },
            })
            .collect();
        let field_host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::RecoverField,
                field_workload_inputs,
                vec![AnalysisNumericalObjectKind::Field],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            host.artifact_access.clone(),
        )
        .unwrap();
        let field_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &field_host,
                inputs: &field_inputs,
            })
            .unwrap();
        let recovered_fields = prepare_kernel_output(&field_outputs[0]);
        let recovered = decode_recovered_field_artifact(&recovered_fields).unwrap();
        assert_eq!(recovered.source_result, field_source_result);
        assert!(recovered.fields().iter().any(|field| {
            field.storage().field_id == runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
        }));

        let definitions = recovered
            .fields()
            .iter()
            .map(|field| field.definition().clone())
            .collect::<Vec<_>>();
        let observation_plan =
            runmat_analysis_fea::results::observation::resolve_structural_observation_plan(
                &solve.model.sequence.steps[0],
                &solve.model,
                Some(&SolverMeshArtifact::canonical_decode(&field_inputs[3].payload()).unwrap()),
                &definitions,
                runmat_analysis_fea::results::observation::ObservationLimits::default(),
            )
            .unwrap();
        let field_selection = RecoveredFieldSelection::new(
            recovered.manifest().unwrap(),
            recovered
                .fields()
                .iter()
                .map(RecoveredField::identity)
                .collect(),
        )
        .unwrap();
        let observation_request = ObservationEvaluationRequest::new(
            step,
            &field_selection,
            executed_constraint_plan.topology,
            observation_plan,
        )
        .unwrap();
        let source_observation_request = observation_request;
        let reference_catalog = RecoveredFieldCatalog::new_for_topology(
            recovered.topology_digest,
            recovered.entity_counts,
            recovered.fields().to_vec(),
        )
        .unwrap();
        let reference_result =
            runmat_analysis_fea::results::observation::evaluate_structural_observations(
                &source_observation_request.plan,
                &solve.model,
                Some(&SolverMeshArtifact::canonical_decode(&field_inputs[3].payload()).unwrap()),
                &reference_catalog,
            )
            .unwrap()
            .remove(0);
        let partition_plan = ObservationPartitionPlan::derive(
            &source_observation_request.plan,
            &source_observation_request.plan.observations[0]
                .spec
                .observation_id,
            solve.observation_execution,
        )
        .unwrap();
        let displacement = recovered
            .fields()
            .iter()
            .find(|field| {
                field.definition().identity()
                    == source_observation_request.plan.observations[0]
                        .source_field_definition_identity
            })
            .unwrap();
        let source_request_objects = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationEvaluationRequest,
            source_observation_request.identity().digest(),
            &source_observation_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let plan_objects = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationPartitionPlan,
            partition_plan.identity.digest(),
            &partition_plan.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let solver_mesh = SolverMeshArtifact::canonical_decode(&field_inputs[3].payload()).unwrap();
        let resource_plan = FeaResourcePlan::from_canonical_bytes(&inputs[3].payload()).unwrap();
        let resource_plan_objects = inputs[3].clone();
        let mut alternate_resource_input = resource_plan.input.clone();
        alternate_resource_input.worker.cpu_millicores += 1;
        let alternate_resource_plan =
            runmat_analysis_fea::planning::resource::build_resource_plan(alternate_resource_input)
                .unwrap();
        assert_ne!(alternate_resource_plan.identity, resource_plan.identity);
        let alternate_resource_plan_objects = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourcePlan,
            alternate_resource_plan.identity.digest(),
            &alternate_resource_plan.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut contribution_objects = Vec::new();
        for partition in &partition_plan.partitions {
            let source_fields = runmat_analysis_fea::results::observation::prepare_structural_observation_field_partition(
                &source_observation_request.plan,
                &partition_plan,
                partition,
                &solve.model,
                Some(&solver_mesh),
                displacement,
            )
            .unwrap();
            let partition_request = ObservationPartitionEvaluationRequest::new(
                &source_observation_request,
                &partition_plan,
                partition,
                &source_fields,
                &resource_plan,
            )
            .unwrap();
            let partition_request_objects = runmat_analysis_execution::prepare_analysis_objects(
                AnalysisNumericalObjectKind::ObservationPartitionEvaluationRequest,
                partition_request.identity().digest(),
                &partition_request.canonical_bytes().unwrap(),
                4096,
                ObjectNamespace::InputValue,
                ObjectInventoryLimits::default(),
            )
            .unwrap();
            let field_partition_objects = runmat_analysis_execution::prepare_analysis_objects(
                AnalysisNumericalObjectKind::FieldPartition,
                source_fields.identity.digest(),
                &source_fields.canonical_bytes().unwrap(),
                4096,
                ObjectNamespace::InputValue,
                ObjectInventoryLimits::default(),
            )
            .unwrap();
            let partition_inputs = vec![
                partition_request_objects,
                source_request_objects.clone(),
                plan_objects.clone(),
                field_partition_objects,
                field_inputs[1].clone(),
                field_inputs[3].clone(),
                resource_plan_objects.clone(),
            ];
            let partition_host = test_host_for_inputs(
                &field_host,
                AnalysisWorkKind::EvaluateObservationPartition,
                &partition_inputs,
                vec![AnalysisNumericalObjectKind::ObservationPartitionContribution],
                None,
            );
            let mut substituted_resource_inputs = partition_inputs.clone();
            *substituted_resource_inputs.last_mut().unwrap() =
                alternate_resource_plan_objects.clone();
            assert!(NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: AnalysisWorkKind::EvaluateObservationPartition,
                    host: &partition_host,
                    inputs: &substituted_resource_inputs,
                })
                .is_err());
            let outputs = NativeAnalysisKernel
                .execute(AnalysisKernelInvocation {
                    kind: AnalysisWorkKind::EvaluateObservationPartition,
                    host: &partition_host,
                    inputs: &partition_inputs,
                })
                .unwrap();
            contribution_objects.push(prepare_kernel_output(&outputs[0]));
        }
        let contributions = contribution_objects
            .iter()
            .map(|objects| {
                ObservationPartitionContribution::from_canonical_bytes(&objects.payload()).unwrap()
            })
            .collect::<Vec<_>>();
        let reduction_request = ObservationPartitionReductionRequest::new(
            &source_observation_request,
            &partition_plan,
            &contributions,
            &resource_plan,
        )
        .unwrap();
        let reduction_request_objects = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationPartitionReductionRequest,
            reduction_request.identity().digest(),
            &reduction_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut reduction_inputs = vec![
            reduction_request_objects,
            source_request_objects,
            plan_objects,
        ];
        reduction_inputs.extend(contribution_objects);
        reduction_inputs.push(resource_plan_objects.clone());
        let reduction_host = test_host_for_inputs(
            &field_host,
            AnalysisWorkKind::ReduceObservationPartitions,
            &reduction_inputs,
            vec![AnalysisNumericalObjectKind::ObservationResult],
            None,
        );
        let mut substituted_reduction_resources = reduction_inputs.clone();
        *substituted_reduction_resources.last_mut().unwrap() =
            alternate_resource_plan_objects.clone();
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::ReduceObservationPartitions,
                host: &reduction_host,
                inputs: &substituted_reduction_resources,
            })
            .is_err());
        let reduction_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::ReduceObservationPartitions,
                host: &reduction_host,
                inputs: &reduction_inputs,
            })
            .unwrap();
        let distributed_result =
            runmat_analysis_fea::results::observation::ObservationResult::from_canonical_bytes(
                &reduction_outputs[0].payload,
            )
            .unwrap();
        assert_eq!(
            distributed_result.logical_identity,
            reference_result.logical_identity
        );
        assert_eq!(distributed_result.partition_evidence.contributions.len(), 2);
        let assembly_request = ObservationAssemblyRequest::new(
            &source_observation_request,
            std::slice::from_ref(&distributed_result),
            &resource_plan,
        )
        .unwrap();
        let assembly_request_objects = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationAssemblyRequest,
            assembly_request.identity().digest(),
            &assembly_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let distributed_result_objects = prepare_kernel_output(&reduction_outputs[0]);
        let assembly_inputs = vec![
            assembly_request_objects,
            reduction_inputs[1].clone(),
            distributed_result_objects,
            resource_plan_objects,
        ];
        let assembly_host = test_host_for_inputs(
            &field_host,
            AnalysisWorkKind::AssembleObservations,
            &assembly_inputs,
            vec![AnalysisNumericalObjectKind::Observation],
            None,
        );
        let mut substituted_assembly_resources = assembly_inputs.clone();
        *substituted_assembly_resources.last_mut().unwrap() = alternate_resource_plan_objects;
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::AssembleObservations,
                host: &assembly_host,
                inputs: &substituted_assembly_resources,
            })
            .is_err());
        let assembly_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::AssembleObservations,
                host: &assembly_host,
                inputs: &assembly_inputs,
            })
            .unwrap();
        let distributed_artifact_objects = prepare_kernel_output(&assembly_outputs[0]);
        let distributed_manifest = ObservationArtifactManifest::from_canonical_bytes(
            &distributed_artifact_objects
                .partition_payload(distributed_artifact_objects.manifest.logical_identity)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(distributed_manifest.results.len(), 1);
        assert_eq!(
            distributed_manifest.results[0].result,
            reference_result.logical_identity
        );
        let observation_manifest = distributed_manifest;

        let accuracy_request = AccuracyEvaluationRequest::new(
            step,
            field_source_result,
            Some(&observation_manifest),
            solve.formulation_requirement,
            solve.model.sequence.steps[0].accuracy.clone(),
        )
        .unwrap();
        let accuracy_request = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::AccuracyEvaluationRequest,
            accuracy_request.identity().digest(),
            &accuracy_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let observation_artifact = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationCatalog,
            observation_manifest.identity.digest(),
            &observation_manifest.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let accuracy_inputs = vec![accuracy_request, numerical_receipt, observation_artifact];
        let accuracy_workload_inputs = accuracy_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if index == 0 || index == 2 {
                    ValueRefKind::DriverObject
                } else {
                    ValueRefKind::ResultObject
                },
            })
            .collect();
        let accuracy_host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::EvaluateAccuracy,
                accuracy_workload_inputs,
                vec![AnalysisNumericalObjectKind::AccuracyReport],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            host.artifact_access.clone(),
        )
        .unwrap();
        let accuracy_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::EvaluateAccuracy,
                host: &accuracy_host,
                inputs: &accuracy_inputs,
            })
            .unwrap();
        let accuracy =
            AccuracyArtifact::from_canonical_bytes(&accuracy_outputs[0].payload).unwrap();
        assert_eq!(
            accuracy.evaluation().report().status,
            AccuracyStatus::Satisfied
        );
        assert_eq!(accuracy.gate().decision, PublicationDecision::Allow);
        assert_eq!(accuracy.source_result, field_source_result);

        let publication_request =
            ResultPublicationRequest::new(&accuracy, Some(&observation_manifest)).unwrap();
        let publication_request = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::PublicationRequest,
            publication_request.identity().digest(),
            &publication_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let accuracy_artifact = runmat_analysis_execution::prepare_analysis_objects(
            accuracy_outputs[0].kind,
            accuracy_outputs[0].logical_identity,
            &accuracy_outputs[0].payload,
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let allocation = prepared.resource_plan.request.clone();
        let resource_usage = record_resource_usage(
            &prepared.resource_plan,
            host.workload.identity.digest(),
            field_source_result.digest(),
            runmat_execution::identity::AttemptId::derive(&[b"kernel-attempt"]),
            runmat_execution::identity::WorkerId::derive(&[b"kernel-worker"]),
            allocation.clone(),
            AnalysisResourceUsage {
                scheduler: ResourceUsage {
                    cpu_core_millis: 1,
                    memory_byte_millis: u128::from(allocation.memory_bytes),
                    wall_millis: 1,
                    ..ResourceUsage::default()
                },
                peak_cpu_millicores: allocation.cpu_millicores,
                peak_memory_bytes: ByteCount(allocation.memory_bytes),
                peak_scratch_bytes: ByteCount(allocation.scratch_bytes),
                artifact_bytes: ByteCount(0),
            },
        )
        .unwrap();
        let resource_usage = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ResourceUsage,
            resource_usage.identity().digest(),
            &resource_usage.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let publication_inputs = vec![
            publication_request,
            accuracy_artifact,
            inputs[3].clone(),
            resource_usage,
            accuracy_inputs[2].clone(),
        ];
        let publication_workload_inputs = publication_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if index == 0 || index == 2 || index == 4 {
                    ValueRefKind::DriverObject
                } else {
                    ValueRefKind::ResultObject
                },
            })
            .collect();
        let publication_host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::PublishResult,
                publication_workload_inputs,
                vec![AnalysisNumericalObjectKind::PublishedResult],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            host.artifact_access.clone(),
        )
        .unwrap();
        let publication_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::PublishResult,
                host: &publication_host,
                inputs: &publication_inputs,
            })
            .unwrap();
        let published =
            PublishedAnalysisResult::from_canonical_bytes(&publication_outputs[0].payload).unwrap();
        assert_eq!(published.source_result, field_source_result);
        assert_eq!(
            published.source_observations,
            Some(observation_manifest.identity)
        );
        assert_eq!(published.source_accuracy, accuracy.identity());

        let mut substituted_accuracy_inputs = accuracy_inputs.clone();
        substituted_accuracy_inputs[2].manifest.logical_identity =
            digest("substituted-observations");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::EvaluateAccuracy,
                host: &accuracy_host,
                inputs: &substituted_accuracy_inputs,
            })
            .is_err());

        let mut substituted_field_inputs = field_inputs.clone();
        substituted_field_inputs[2].manifest.logical_identity = digest("substituted-mesh");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &field_host,
                inputs: &substituted_field_inputs,
            })
            .is_err());

        let publication = SemanticStatePublicationRequest::structural_linear(
            step,
            solve.model.sequence.steps[0].state_outputs[0].clone(),
            inputs[1].manifest.logical_identity,
            SolverResultIdentity::new(
                host.workload.identity,
                solver_result.manifest.payload_digest,
            )
            .unwrap(),
        )
        .unwrap();
        let publication_bytes = publication.canonical_bytes().unwrap();
        let publication = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::StatePublicationRequest,
            Digest::sha256(&publication_bytes),
            &publication_bytes,
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let state_inputs = vec![publication, solver_result];
        let state_workload_inputs = state_inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: ValueRefKind::DriverObject,
            })
            .collect();
        let state_host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::PublishState,
                state_workload_inputs,
                vec![AnalysisNumericalObjectKind::SemanticState],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            host.artifact_access.clone(),
        )
        .unwrap();
        let state_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::PublishState,
                host: &state_host,
                inputs: &state_inputs,
            })
            .unwrap();
        let state =
            CanonicalSemanticState::from_canonical_bytes(&state_outputs[0].payload).unwrap();
        assert_eq!(
            state.fields[0].field_id,
            runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
        );
        assert_eq!(state.topology_digest, inputs[1].manifest.logical_identity);

        let consumer_step = solve
            .model
            .sequence
            .steps
            .iter()
            .find(|candidate| candidate.step_id.as_str() == "branch")
            .unwrap();
        let mut warm_solve = solve.clone();
        warm_solve.step_id = consumer_step.step_id.clone();
        let warm_state = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SemanticState,
            state.identity.digest(),
            &state.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let warm_prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
            authored_study_digest: digest("study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: host.artifact_access.clone(),
            workload: warm_solve,
            mesh: Some(SolverMeshArtifact::canonical_decode(&inputs[1].payload()).unwrap()),
            initial_guess: Some(warm_state),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        })
        .unwrap();
        let warm_host = warm_prepared.host;
        let warm_inputs = warm_prepared.inputs;
        let warm_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &warm_host,
                inputs: &warm_inputs,
            })
            .unwrap();
        let warm_result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&warm_outputs[0].payload).unwrap();
        assert_eq!(
            result
                .run
                .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT),
            warm_result
                .run
                .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        );
        assert!(warm_result.run.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "FEA_CONVERGENCE" && diagnostic.message.contains("iterations=0 ")
        }));

        let mut substituted = warm_inputs.clone();
        substituted[4].manifest.logical_identity = digest("substituted-state");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &warm_host,
                inputs: &substituted,
            })
            .is_err());
    }

    #[test]
    fn native_kernel_executes_explicit_structural_workload_without_mesh_input() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.sequence.steps[0].observations = vec![structural_displacement_observation()];
        let solve = StructuralLinearWorkload {
            schema_version: 1,
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            require_solver_mesh_for_solid: true,
            initial_guess_transition: None,
        };
        let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
            authored_study_digest: digest("explicit-study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "study:explicit".into(),
                encryption_context: digest("explicit-encryption"),
            },
            workload: solve.clone(),
            mesh: None,
            initial_guess: None,
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        })
        .unwrap();
        assert_eq!(prepared.inputs.len(), 3);
        assert!(prepared
            .inputs
            .iter()
            .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh));

        let outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &prepared.host,
                inputs: &prepared.inputs,
            })
            .unwrap();
        let result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&outputs[0].payload).unwrap();
        assert!(matches!(
            result.constraint_result.topology,
            CanonicalStructuralTopologyIdentity::Explicit { .. }
        ));
        assert!(result.run.fields.iter().any(|field| {
            field.field_id == runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
                && field.as_host_f64().is_some_and(|values| {
                    !values.is_empty() && values.iter().all(|value| value.is_finite())
                })
        }));

        let solver_result = prepare_kernel_output(&outputs[0]);
        let source_result = SolverResultIdentity::new(
            prepared.host.workload.identity,
            solver_result.manifest.payload_digest,
        )
        .unwrap();
        let recovery_request = FieldRecoveryRequest::new(
            prepared.host.workload.step,
            source_result,
            result.constraint_result.topology,
        )
        .unwrap();
        let recovery_request = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::FieldRecoveryRequest,
            recovery_request.identity().digest(),
            &recovery_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let recovery_inputs = vec![recovery_request, prepared.inputs[0].clone(), solver_result];
        assert!(recovery_inputs
            .iter()
            .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh));
        let recovery_host = test_host_for_inputs(
            &prepared.host,
            AnalysisWorkKind::RecoverField,
            &recovery_inputs,
            vec![AnalysisNumericalObjectKind::Field],
            Some(2),
        );
        let recovered_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &recovery_host,
                inputs: &recovery_inputs,
            })
            .unwrap();
        let recovered_objects = prepare_kernel_output(&recovered_outputs[0]);
        let recovered = decode_recovered_field_artifact(&recovered_objects).unwrap();
        assert_eq!(
            recovered.topology_digest,
            result.constraint_result.topology.discretization_digest()
        );
        assert_eq!(recovered.entity_counts.solid_elements, 0);
        assert!(recovered.entity_counts.beam_elements > 0);

        let mut substituted_workload = solve.clone();
        substituted_workload.model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
        substituted_workload.step_id = substituted_workload.model.sequence.steps[0].step_id.clone();
        let substituted_invocation = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SolverInvocation,
            digest("substituted-explicit-invocation"),
            &substituted_workload.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut substituted_recovery_inputs = recovery_inputs.clone();
        substituted_recovery_inputs[1] = substituted_invocation.clone();
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::RecoverField,
                host: &recovery_host,
                inputs: &substituted_recovery_inputs,
            })
            .is_err());
    }

    #[test]
    fn native_kernel_executes_conformal_mixed_structural_workload() {
        use runmat_analysis_core::{
            AnalysisConstraint, BeamElementModel, BeamSectionModel, ConformalSolidAttachment,
            CoordinateFrame, PrescribedStructuralDofs, PrescribedValue, RegionTarget, Rotation,
            RotationDofValue, StructuralConstraint, StructuralConstraintKind, StructuralElement,
            StructuralElementKind, StructuralModel, StructuralNode, StructuralRotationDof,
        };

        let mut mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let attached_coordinates = mesh.topology.nodes[0].coordinates_m;
        mesh.topology.nodes[0].provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Vertex,
            source_topology_id: "beam_root".into(),
            assembly_path: vec!["root".into()],
        });
        for (face_index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
            face.provenance.push(PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: if face_index == 0 { "root" } else { "tip" }.into(),
                assembly_path: vec!["root".into()],
            });
            face.provenance.sort();
        }
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let assignment = model.material_assignments.first().unwrap();
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: assignment.region_id.clone(),
            assembly_path: vec!["root".into()],
        };
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();
        model.geometry_revision = mesh.geometry.geometry_revision as u32;
        model.structural = Some(StructuralModel {
            nodes: vec![
                StructuralNode {
                    node_id: 101,
                    coordinates_m: attached_coordinates,
                    solid_attachment: Some(ConformalSolidAttachment {
                        geometry_vertex_id: "beam_root".into(),
                    }),
                },
                StructuralNode {
                    node_id: 102,
                    coordinates_m: [
                        attached_coordinates[0] + 1.0,
                        attached_coordinates[1],
                        attached_coordinates[2],
                    ],
                    solid_attachment: None,
                },
            ],
            elements: vec![StructuralElement {
                element_id: "beam".into(),
                region_id: "beam_span".into(),
                kind: StructuralElementKind::Beam(BeamElementModel {
                    node_ids: [101, 102],
                    section_id: "beam_section".into(),
                    reference_axis: [0.0, 1.0, 0.0],
                }),
            }],
            beam_sections: vec![BeamSectionModel {
                section_id: "beam_section".into(),
                area_m2: 1.0e-4,
                iy_m4: 1.0e-9,
                iz_m4: 1.0e-9,
                torsion_j_m4: 1.0e-9,
                outer_fiber_y_m: 0.01,
                outer_fiber_z_m: 0.01,
                torsion_outer_radius_m: 0.01,
            }],
            shell_sections: Vec::new(),
        });
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: "beam_root_rotation".into(),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point("node:101"),
                    translations: Vec::new(),
                    rotations: [
                        StructuralRotationDof::Rx,
                        StructuralRotationDof::Ry,
                        StructuralRotationDof::Rz,
                    ]
                    .into_iter()
                    .map(|dof| RotationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Rotation::new(0.0)),
                    })
                    .collect(),
                }),
            }));
        let solve = StructuralLinearWorkload {
            schema_version: 1,
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            require_solver_mesh_for_solid: true,
            initial_guess_transition: None,
        };
        let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
            authored_study_digest: digest("mixed-study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "study:mixed".into(),
                encryption_context: digest("mixed-encryption"),
            },
            workload: solve,
            mesh: Some(mesh),
            initial_guess: None,
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        })
        .unwrap();
        let outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &prepared.host,
                inputs: &prepared.inputs,
            })
            .unwrap();
        let result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&outputs[0].payload).unwrap();
        assert!(result
            .run
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity
                != runmat_analysis_fea::diagnostics::FeaDiagnosticSeverity::Error));
        assert_eq!(
            result
                .run
                .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .shape,
            vec![5, 3]
        );
        let plan =
            CanonicalStructuralConstraintPlan::from_canonical_bytes(&prepared.inputs[2].payload())
                .unwrap();
        assert_eq!(result.constraint_result.topology, plan.topology);
        result.constraint_result.validate_against(&plan).unwrap();
        let rotation_group = result
            .constraint_result
            .groups
            .iter()
            .find(|group| group.constraint_id == "beam_root_rotation")
            .unwrap();
        assert!(rotation_group.row_reactions.iter().all(|row| matches!(
            row.violation,
            runmat_analysis_fea::constraints::ConstraintViolation::RotationRadians(_)
        )));
        let mut substituted = result.constraint_result.clone();
        let CanonicalStructuralTopologyIdentity::ConformalMixed {
            structural_topology_digest,
            ..
        } = &mut substituted.topology
        else {
            panic!("mixed solve must retain mixed topology identity");
        };
        *structural_topology_digest = digest("substituted-structural-topology");
        assert!(substituted.validate_against(&plan).is_err());
    }

    fn structural_displacement_observation() -> runmat_analysis_core::ObservationRequest {
        runmat_analysis_core::ObservationRequest::Field(runmat_analysis_core::ObservationSpec {
            observation_id: runmat_analysis_core::ObservationId("whole_displacement".into()),
            quantity: runmat_analysis_core::ObservationQuantity::Displacement,
            value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
            target: runmat_analysis_core::ObservationTarget::WholeModel,
            source_location: runmat_analysis_core::FieldLocation::Node,
            output_location: runmat_analysis_core::FieldLocation::Node,
            recovery: runmat_analysis_core::RecoveryPolicy::Native,
            component: runmat_analysis_core::ObservationComponent::All,
            frame: runmat_analysis_core::CoordinateFrame::Global,
            sign_convention: runmat_analysis_core::SignConvention::Native,
            complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
            sampling: runmat_analysis_core::SamplingPolicy::Final,
            reduction: None,
            storage: runmat_analysis_core::ObservationStoragePolicy {
                precision: runmat_analysis_core::SignalPrecision::Real64,
                compression: runmat_analysis_core::CompressionPolicy::Lossless,
                retention: runmat_analysis_core::RetentionPolicy::Canonical,
                partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: runmat_analysis_core::ObservationPurpose::Publication,
            cost: runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000,
                peak_resident_bytes: 1_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            },
        })
    }
}
