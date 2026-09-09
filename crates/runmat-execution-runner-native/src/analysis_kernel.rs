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
    ObservationEvaluationRequest, ResultPublicationRequest, SemanticStatePublicationRequest,
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
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldCatalog, RecoveredFieldSelection,
};
use runmat_analysis_fea::results::observation::{
    evaluate_observations, resolve_observation_plan, ObservationArtifact, ObservationArtifactInput,
    ObservationArtifactManifest, ObservationLimits,
};
use runmat_analysis_fea::results::publication::PublishedAnalysisResult;
use runmat_analysis_fea::topology::AnalysisProjection;
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
            AnalysisWorkKind::EvaluateObservation => execute_observation_evaluation(invocation),
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

fn execute_observation_evaluation(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    let request = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::ObservationEvaluationRequest,
    )?;
    let fields = unique_input(&invocation, AnalysisNumericalObjectKind::FieldSelection)?;
    let mesh = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    if invocation.inputs.len() != 3
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::Observation]
    {
        return Err(AnalysisBridgeError::Invalid(
            "observation evaluation requires one request, field artifact, mesh, and observation output"
                .into(),
        ));
    }
    let request_manifest_identity = request.manifest.logical_identity;
    let field_selection_identity = fields.manifest.logical_identity;
    let request = ObservationEvaluationRequest::from_canonical_bytes(&request.payload())?;
    let selection = RecoveredFieldSelection::from_canonical_bytes(
        &fields.partition_payload(field_selection_identity)?,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let selected_fields = selection
        .fields
        .iter()
        .map(|identity| {
            RecoveredField::from_canonical_bytes(&fields.partition_payload(identity.digest())?)
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    if request.identity().digest() != request_manifest_identity
        || request.producer != invocation.host.workload.step
        || selection.identity.digest() != field_selection_identity
        || request.source_selection != selection.identity
        || request.source_fields != selection.source.identity
        || request.field_recovery_request != selection.source.recovery_request
        || request.source_result != selection.source.source_result
        || request.mesh_digest != mesh.manifest.logical_identity
        || request.plan.topology_digest != selection.source.topology_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "observation inputs differ from the admitted evaluation request",
        ));
    }
    let mesh = SolverMeshArtifact::canonical_decode(&mesh.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()) != request.mesh_digest {
        return Err(AnalysisBridgeError::Identity(
            "observation mesh content differs from its admitted identity",
        ));
    }
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let catalog = RecoveredFieldCatalog::new(&projection, selected_fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let results = evaluate_observations(&request.plan, &projection, &catalog)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: request.identity(),
        producer: request.producer,
        field_recovery_request: request.field_recovery_request,
        source_fields: request.source_fields,
        source_result: request.source_result,
        topology_digest: request.plan.topology_digest,
        plan_identity: request.plan.identity,
        results,
    })
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
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
    let solver_result = unique_input(&invocation, AnalysisNumericalObjectKind::SolverResult)?;
    let mesh = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    if invocation.inputs.len() != 3
        || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::Field]
    {
        return Err(AnalysisBridgeError::Invalid(
            "field recovery requires one request, solver result, mesh, and field output".into(),
        ));
    }
    let request_manifest_identity = request.manifest.logical_identity;
    let request = FieldRecoveryRequest::from_canonical_bytes(&request.payload())?;
    if request.identity().digest() != request_manifest_identity
        || request.producer != invocation.host.workload.step
        || request.source_result.digest() != solver_result.manifest.logical_identity
        || request.mesh_digest != mesh.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "field recovery inputs differ from the admitted source request",
        ));
    }
    let mesh = SolverMeshArtifact::canonical_decode(&mesh.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()) != request.mesh_digest {
        return Err(AnalysisBridgeError::Identity(
            "field recovery mesh content differs from its topology identity",
        ));
    }
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if projection.projection_digest != request.projection_digest {
        return Err(AnalysisBridgeError::Identity(
            "field recovery projection differs from its admitted identity",
        ));
    }
    let mut solver_result =
        StructuralLinearWorkloadResult::from_canonical_bytes(&solver_result.payload())?;
    let fields = take_structural_observable_fields(&mut solver_result.run.fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        request.source_result,
        request.projection_digest,
        mesh.topology.nodes.len(),
        mesh.topology.volume_elements.len(),
        fields,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let manifest = artifact
        .manifest()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mut partitions = Vec::with_capacity(artifact.fields().len() + 1);
    partitions.push(AnalysisKernelOutputPartition {
        logical_identity: artifact.identity().digest(),
        payload: manifest
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    });
    for field in artifact.fields() {
        partitions.push(AnalysisKernelOutputPartition {
            logical_identity: field.identity().digest(),
            payload: field
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        });
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
    let mesh = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let constraint_plan = unique_input(&invocation, AnalysisNumericalObjectKind::ConstraintPlan)?;
    let resource_plan_input = unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?;
    let solve = StructuralLinearWorkload::from_canonical_bytes(&solve_invocation.payload())?;
    let expected_inputs = if solve.initial_guess_transition.is_some() {
        5
    } else {
        4
    };
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
                || state.topology_digest != mesh.manifest.logical_identity
                || transition.target_topology_digest != mesh.manifest.logical_identity
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
    let solver_model = solve.solver_model()?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&constraint_plan.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    constraint_plan
        .validate_against(&solver_model, &mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&resource_plan_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if resource_plan_input.manifest.logical_identity != resource_plan.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "resource-plan manifest differs from its canonical identity",
        ));
    }
    let projection = AnalysisProjection::derive(&mesh)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let field_definitions = structural_observable_field_definitions()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let observations = resolve_observation_plan(
        solve.procedure_step()?,
        &projection,
        &field_definitions,
        ObservationLimits::default(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let expected_resource_plan =
        build_structural_resource_plan(&projection, &constraint_plan, &observations, &solve)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if expected_resource_plan != resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "resource plan does not converge from authoritative structural inputs",
        ));
    }
    let expected_constraint_identity = ConstraintPlanIdentity::new(
        invocation.host.workload.step,
        runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
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
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
            thermo_mechanical_context: solve.thermo_mechanical_context,
            electro_thermal_context: solve.electro_thermal_context,
            initial_displacement,
        },
        &constraint_plan,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let constraint_result =
        CanonicalStructuralConstraintResult::recover(&constraint_plan, &mesh, &run.fields)
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
    use runmat_analysis_fea::execution::SolverResultIdentity;
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::planning::resource::{
        record_resource_usage, AnalysisResourceUsage, ByteCount,
    };
    use runmat_analysis_fea::results::accuracy::{AccuracyStatus, PublicationDecision};
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
            require_solver_mesh_for_solid: false,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
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
            mesh,
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
        let field_recovery_request = FieldRecoveryRequest::new(
            step,
            field_source_result,
            inputs[1].manifest.logical_identity,
            runmat_analysis_fea::topology::AnalysisProjection::derive(
                &SolverMeshArtifact::canonical_decode(&inputs[1].payload()).unwrap(),
            )
            .unwrap()
            .projection_digest,
        )
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
                reference_kind: if index == 1 {
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

        let observation_projection = runmat_analysis_fea::topology::AnalysisProjection::derive(
            &SolverMeshArtifact::canonical_decode(&field_inputs[2].payload()).unwrap(),
        )
        .unwrap();
        let definitions = recovered
            .fields()
            .iter()
            .map(|field| field.definition().clone())
            .collect::<Vec<_>>();
        let observation_plan = runmat_analysis_fea::results::observation::resolve_observation_plan(
            &solve.model.sequence.steps[0],
            &observation_projection,
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
            field_inputs[2].manifest.logical_identity,
            observation_plan,
        )
        .unwrap();
        let observation_request = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationEvaluationRequest,
            observation_request.identity().digest(),
            &observation_request.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut field_selection_partitions = vec![AnalysisKernelOutputPartition {
            logical_identity: field_selection.identity.digest(),
            payload: field_selection.canonical_bytes().unwrap(),
        }];
        field_selection_partitions.extend(recovered.fields().iter().map(|field| {
            AnalysisKernelOutputPartition {
                logical_identity: field.identity().digest(),
                payload: field.canonical_bytes().unwrap(),
            }
        }));
        let recovered_fields = prepare_kernel_output(&AnalysisKernelOutput::partitioned(
            AnalysisNumericalObjectKind::FieldSelection,
            field_selection.identity.digest(),
            field_selection_partitions,
        ));
        let observation_inputs = vec![
            observation_request,
            recovered_fields,
            field_inputs[2].clone(),
        ];
        let observation_workload_inputs = observation_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if index == 1 {
                    ValueRefKind::ResultObject
                } else {
                    ValueRefKind::DriverObject
                },
            })
            .collect();
        let observation_host = AnalysisHostWorkload::new(
            AnalysisWorkloadRequest::new(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::EvaluateObservation,
                observation_workload_inputs,
                vec![AnalysisNumericalObjectKind::Observation],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap(),
            host.artifact_access.clone(),
        )
        .unwrap();
        let observation_outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::EvaluateObservation,
                host: &observation_host,
                inputs: &observation_inputs,
            })
            .unwrap();
        let observation_objects = prepare_kernel_output(&observation_outputs[0]);
        let observation_manifest = ObservationArtifactManifest::from_canonical_bytes(
            &observation_objects
                .partition_payload(observation_objects.manifest.logical_identity)
                .unwrap(),
        )
        .unwrap();
        let observation_results = observation_manifest
            .results
            .iter()
            .map(|entry| {
                runmat_analysis_fea::results::observation::ObservationResult::from_canonical_bytes(
                    &observation_objects
                        .partition_payload(entry.result.digest())
                        .unwrap(),
                )
                .unwrap()
            })
            .collect();
        let observation = ObservationArtifact::from_manifest_and_results(
            observation_manifest.clone(),
            observation_results,
        )
        .unwrap();
        assert_eq!(observation.results().len(), 1);
        assert_eq!(
            observation.results()[0].observation_id.0,
            "whole_displacement"
        );

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

        let mut substituted_observation_inputs = observation_inputs.clone();
        substituted_observation_inputs[1].manifest.logical_identity = digest("substituted-fields");
        assert!(NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::EvaluateObservation,
                host: &observation_host,
                inputs: &substituted_observation_inputs,
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
            mesh: SolverMeshArtifact::canonical_decode(&inputs[1].payload()).unwrap(),
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
