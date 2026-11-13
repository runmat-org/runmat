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
    protocol, ExactStructuralModalResponseSpectrumAnalysisResult,
    ExactStructuralStudyAnalysisResult,
};
use crate::NativeExecutionResult;

pub(super) struct ResponseSpectrumResultImport<'a, Source> {
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
    pub field_recovery_root: ValueRef,
    pub envelope_state_root: ValueRef,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a SolverMeshArtifact,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_response_spectrum_result<Source: CacheImport>(
    input: ResponseSpectrumResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let authored = runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: input.workloads,
    };
    let step_ids = authored.step_ids().map_err(protocol)?;
    let [modal_step, response_step] = step_ids.as_slice() else {
        return Err(protocol(
            "modal response-spectrum result requires exactly two authored steps",
        ));
    };
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        &authored
    else {
        return Err(protocol(
            "response-spectrum workload inventory changed kind",
        ));
    };
    let StructuralStepWorkload::ModalExtraction {
        workload: modal_workload,
    } = &workloads[modal_step]
    else {
        return Err(protocol(
            "response-spectrum producer is not modal extraction",
        ));
    };
    let StructuralStepWorkload::ModalResponseSpectrum { template } = &workloads[response_step]
    else {
        return Err(protocol(
            "response-spectrum consumer is not a response template",
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
            &input.field_recovery_root,
            AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
        ),
        (
            &input.envelope_state_root,
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
    let [modal_result_objects, modal_receipt_objects, basis_state_objects, response_result_objects, response_receipt_objects, response_accuracy_objects, response_publication_baseline_usage_objects, response_publication_objects, response_resource_usage_objects, field_recovery_objects, envelope_state_objects] =
        imported.as_slice()
    else {
        return Err(protocol("response-spectrum result closure is incomplete"));
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
            "published modal basis differs from its response-spectrum producer",
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
        runmat_analysis_execution::decode_prepared_structural_modal_response_spectrum_solver_result(
            response_result_objects,
        )
        .map_err(protocol)?;
    response_result
        .validate_for(&response_workload)
        .map_err(protocol)?;
    let response_numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_modal_response_spectrum_numerical_receipt(
            response_receipt_objects,
        )
        .map_err(protocol)?;
    if response_result_objects.manifest.logical_identity
        != response_numerical_receipt.source_result().digest()
    {
        return Err(protocol(
            "response-spectrum result object differs from its numerical receipt source",
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
            "remote response-spectrum publication does not yet package authored observations",
        ));
    }
    let accuracy_request = runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
        producer,
        runmat_analysis_fea::execution::AccuracySourceKind::StructuralModalResponseSpectrum,
        response_numerical_receipt.source_result(),
        None,
        response_workload
            .procedure_step()
            .map_err(protocol)?
            .accuracy
            .clone(),
    )
    .map_err(protocol)?;
    let source =
        runmat_analysis_fea::execution::AccuracySourceReceipt::StructuralModalResponseSpectrum(
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
            "response-spectrum accuracy differs from its exact numerical source",
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
        .ok_or_else(|| protocol("response-spectrum resource evidence omits its primary plan"))?;
    if primary_plan.input.phase
        != runmat_analysis_fea::planning::resource::FeaResourcePhase::StructuralModalResponseSpectrumAssembly(
            response_result.assembly().identity(),
        )
    {
        return Err(protocol(
            "response-spectrum resource evidence names another terminal assembly",
        ));
    }
    let field_recovery =
        runmat_analysis_execution::decode_prepared_structural_modal_response_spectrum_field_recovery(
            field_recovery_objects,
        )
        .map_err(protocol)?;
    field_recovery
        .validate_for_result(response_result.numerical())
        .map_err(protocol)?;
    let constraint_plan =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            response_workload.model(),
            Some(input.mesh),
        )
        .map_err(protocol)?;
    let prepared_response = runmat_analysis_fea::prepare_structural_modal_response_spectrum(
        &response_workload,
        input.mesh,
        &basis_state,
        &constraint_plan,
        &[],
    )
    .map_err(protocol)?;
    let expected = runmat_analysis_fea::recover_structural_modal_response_spectrum_fields(
        &response_workload,
        &prepared_response,
        response_result.numerical(),
    )
    .map_err(protocol)?;
    if field_recovery != expected {
        return Err(protocol(
            "response-spectrum field recovery differs from its exact physical inputs",
        ));
    }
    let envelope_state =
        runmat_analysis_execution::decode_prepared_semantic_state(envelope_state_objects)
            .map_err(protocol)?;
    let expected_envelope =
        runmat_analysis_fea::execution::StructuralResponseEnvelope::from_result(&response_result)
            .map_err(protocol)?;
    if envelope_state.source_result != response_numerical_receipt.source_result()
        || envelope_state.topology_digest != topology_digest
        || envelope_state.payload.structural_response_envelope() != Some(&expected_envelope)
    {
        return Err(protocol(
            "published response envelope differs from its exact response-spectrum result",
        ));
    }
    Ok(
        ExactStructuralStudyAnalysisResult::StructuralModalResponseSpectrum(Box::new(
            ExactStructuralModalResponseSpectrumAnalysisResult {
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
                field_recovery,
                envelope_state,
                // Remote packaging does not carry statistical observation catalogs; steps with
                // authored outputs are rejected below before this result is built.
                response_observations: None,
            },
        )),
    )
}
