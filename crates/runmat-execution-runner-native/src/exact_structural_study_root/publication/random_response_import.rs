use std::collections::BTreeMap;

use runmat_analysis_core::ProcedureStepId;
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, StructuralStepWorkload};
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_meshing_core::SolverMeshArtifact;

use super::modal_import::validate_modal_pair;
use super::result_import::{
    import_numerical_analysis_objects, validate_complete_numerical_inventory,
};
use super::{
    protocol, ExactStructuralRandomResponseAnalysisResult, ExactStructuralStudyAnalysisResult,
};
use crate::NativeExecutionResult;

pub(super) struct RandomResponseResultImport<'a, Source> {
    pub source: &'a Source,
    pub workloads: BTreeMap<ProcedureStepId, StructuralStepWorkload>,
    pub modal_result_root: ValueRef,
    pub modal_receipt_root: ValueRef,
    pub basis_state_root: ValueRef,
    pub response_result_root: ValueRef,
    pub response_receipt_root: ValueRef,
    pub response_accuracy_root: ValueRef,
    pub response_publication_baseline_usage_root: ValueRef,
    pub response_resource_usage_root: ValueRef,
    pub response_publication_root: ValueRef,
    pub statistics_state_root: ValueRef,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a SolverMeshArtifact,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_random_response_result<Source: CacheImport>(
    input: RandomResponseResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let authored = runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: input.workloads,
    };
    let step_ids = authored.step_ids().map_err(protocol)?;
    let [modal_step, response_step] = step_ids.as_slice() else {
        return Err(protocol(
            "random-response result requires exactly two authored steps",
        ));
    };
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        &authored
    else {
        return Err(protocol("random-response workload inventory changed kind"));
    };
    let StructuralStepWorkload::ModalExtraction {
        workload: modal_workload,
    } = &workloads[modal_step]
    else {
        return Err(protocol("random-response producer is not modal extraction"));
    };
    let StructuralStepWorkload::RandomResponse { template } = &workloads[response_step] else {
        return Err(protocol(
            "random-response consumer is not a random-response template",
        ));
    };
    let roots = [
        (
            &input.modal_result_root,
            AnalysisNumericalObjectKind::SolverResult,
        ),
        (
            &input.modal_receipt_root,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ),
        (
            &input.basis_state_root,
            AnalysisNumericalObjectKind::SemanticState,
        ),
        (
            &input.response_result_root,
            AnalysisNumericalObjectKind::SolverResult,
        ),
        (
            &input.response_receipt_root,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ),
        (
            &input.response_accuracy_root,
            AnalysisNumericalObjectKind::AccuracyReport,
        ),
        (
            &input.response_publication_baseline_usage_root,
            AnalysisNumericalObjectKind::ResourceUsage,
        ),
        (
            &input.response_publication_root,
            AnalysisNumericalObjectKind::PublishedResult,
        ),
        (
            &input.response_resource_usage_root,
            AnalysisNumericalObjectKind::ResourceUsage,
        ),
        (
            &input.statistics_state_root,
            AnalysisNumericalObjectKind::SemanticState,
        ),
    ];
    let imported = roots
        .iter()
        .map(|(root, kind)| {
            import_numerical_analysis_objects(
                input.source,
                root,
                &input.objects,
                *kind,
                input.access,
                input.limits,
            )
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    validate_complete_numerical_inventory(
        &imported.iter().collect::<Vec<_>>(),
        &input.objects,
        input.access,
    )?;
    let [modal_result_objects, modal_receipt_objects, basis_state_objects, response_result_objects, response_receipt_objects, response_accuracy_objects, response_publication_baseline_usage_objects, response_publication_objects, response_resource_usage_objects, statistics_state_objects] =
        imported.as_slice()
    else {
        return Err(protocol("random-response result closure is incomplete"));
    };
    let modal_result =
        runmat_analysis_execution::decode_prepared_structural_modal_result(modal_result_objects)
            .map_err(protocol)?;
    modal_result
        .validate_for(modal_workload)
        .map_err(protocol)?;
    let modal_numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_modal_receipt(modal_receipt_objects)
            .map_err(protocol)?;
    validate_modal_pair(
        modal_result_objects,
        &modal_result,
        &modal_numerical_receipt,
    )?;
    let basis_state =
        runmat_analysis_execution::decode_prepared_semantic_state(basis_state_objects)
            .map_err(protocol)?;
    if basis_state.source_result != modal_numerical_receipt.source_result()
        || basis_state
            .payload
            .modal_basis()
            .map(|basis| basis.identity())
            != Some(modal_result.basis_identity())
    {
        return Err(protocol(
            "published modal basis differs from its random-response producer",
        ));
    }
    let topology_digest = Digest::from_bytes(*input.mesh.canonical_digest.bytes());
    let transition = runmat_analysis_execution::bind_upstream_state_transition(
        &template.model,
        &template.step_id,
        &basis_state,
        runmat_analysis_core::StateUse::ModalBasis,
        topology_digest,
    )
    .map_err(protocol)?;
    let response_workload = template.clone().bind(transition).map_err(protocol)?;
    let response_result =
        runmat_analysis_execution::decode_prepared_structural_random_response_solver_result(
            response_result_objects,
        )
        .map_err(protocol)?;
    response_result
        .validate_for(&response_workload)
        .map_err(protocol)?;
    let response_numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_random_response_numerical_receipt(
            response_receipt_objects,
        )
        .map_err(protocol)?;
    if response_result_objects.manifest.logical_identity
        != response_numerical_receipt.source_result().digest()
    {
        return Err(protocol(
            "random-response result object differs from its numerical receipt source",
        ));
    }
    response_numerical_receipt
        .validate_against(response_numerical_receipt.source_result(), &response_result)
        .map_err(protocol)?;
    let response_accuracy =
        runmat_analysis_fea::results::accuracy::AccuracyArtifact::from_canonical_bytes(
            &response_accuracy_objects.payload(),
        )
        .map_err(protocol)?;
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &response_workload.model().sequence,
    )
    .map_err(protocol)?;
    let producer = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
        sequence,
        response_workload.procedure_step().map_err(protocol)?,
    )
    .map_err(protocol)?;
    if !response_workload
        .procedure_step()
        .map_err(protocol)?
        .observations
        .is_empty()
    {
        return Err(protocol(
            "remote random-response publication does not yet package authored observations",
        ));
    }
    let accuracy_request = runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
        producer,
        runmat_analysis_fea::execution::AccuracySourceKind::StructuralRandomResponse,
        response_numerical_receipt.source_result(),
        None,
        response_workload
            .procedure_step()
            .map_err(protocol)?
            .accuracy
            .clone(),
    )
    .map_err(protocol)?;
    let source = runmat_analysis_fea::execution::AccuracySourceReceipt::StructuralRandomResponse(
        response_numerical_receipt.clone(),
    );
    let expected_accuracy = runmat_analysis_fea::results::accuracy::evaluate_accuracy_artifact(
        &accuracy_request,
        source.evidence(&accuracy_request).map_err(protocol)?,
    )
    .map_err(protocol)?;
    if response_accuracy_objects.manifest.logical_identity != response_accuracy.identity().digest()
        || response_accuracy != expected_accuracy
    {
        return Err(protocol(
            "random-response accuracy differs from its exact numerical source",
        ));
    }
    let (response_publication_baseline_usage, response_resource_usage, response_publication) =
        super::specialized_publication_import::import_specialized_publication(
            response_publication_baseline_usage_objects,
            response_resource_usage_objects,
            response_publication_objects,
            response_workload.identity().digest(),
            response_numerical_receipt.source_result().digest(),
            &response_accuracy,
        )?;
    let primary_plan = response_resource_usage
        .evidence
        .plans
        .iter()
        .find(|plan| plan.identity == response_resource_usage.evidence.primary_plan)
        .ok_or_else(|| protocol("random-response resource evidence omits its primary plan"))?;
    if primary_plan.input.phase
        != runmat_analysis_fea::planning::resource::FeaResourcePhase::StructuralRandomResponseAssembly(
            response_result.assembly().identity(),
        )
    {
        return Err(protocol(
            "random-response resource evidence names another terminal assembly",
        ));
    }
    let constraints =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            response_workload.model(),
            Some(input.mesh),
        )
        .map_err(protocol)?;
    let physical = runmat_analysis_fea::prepare_structural_random_response(
        &response_workload,
        input.mesh,
        &basis_state,
        &constraints,
    )
    .map_err(protocol)?;
    if physical.scope() != response_result.assembly().scope() {
        return Err(protocol(
            "random-response result scope differs from its exact physical inputs",
        ));
    }
    for task in response_result.assembly().tasks() {
        let expected = runmat_analysis_fea::execute_structural_random_response_intervals(
            &response_workload,
            &physical,
            task.request(),
        )
        .map_err(protocol)?;
        if &expected != task {
            return Err(protocol(
                "random-response interval differs from its exact physical inputs",
            ));
        }
    }
    let statistics_state =
        runmat_analysis_execution::decode_prepared_semantic_state(statistics_state_objects)
            .map_err(protocol)?;
    let expected_statistics =
        runmat_analysis_fea::execution::StructuralRandomStatistics::from_result(&response_result)
            .map_err(protocol)?;
    if statistics_state.source_result != response_numerical_receipt.source_result()
        || statistics_state.topology_digest != topology_digest
        || statistics_state.payload.structural_random_statistics() != Some(&expected_statistics)
    {
        return Err(protocol(
            "published random statistics differ from the exact response result",
        ));
    }
    Ok(
        ExactStructuralStudyAnalysisResult::StructuralRandomResponse(Box::new(
            ExactStructuralRandomResponseAnalysisResult {
                modal_workload: modal_workload.clone(),
                response_workload: Box::new(response_workload),
                modal_result: Box::new(modal_result),
                modal_numerical_receipt,
                basis_state,
                response_result: Box::new(response_result),
                response_numerical_receipt,
                response_accuracy,
                response_publication_baseline_usage,
                response_resource_usage,
                response_publication,
                statistics_state,
                // Remote packaging does not carry statistical observation catalogs; steps with
                // authored outputs are rejected below before this result is built.
                response_observations: None,
            },
        )),
    )
}
