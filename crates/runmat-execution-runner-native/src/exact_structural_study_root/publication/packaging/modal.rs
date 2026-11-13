//! Modal numerical closure assembly; no physics is recomputed.
use super::*;

pub(super) fn modal_closure(
    completion: &crate::CompletedStructuralModalStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let result_root = completion
        .result
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let receipt_root = completion
        .result
        .numerical_receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let objects = unique_references(
        completion
            .result
            .objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?
            .into_iter()
            .chain(
                completion
                    .result
                    .numerical_receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            ),
    )?;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralModal {
        result_root: Box::new(result_root),
        receipt_root: Box::new(receipt_root),
        objects,
    })
}

pub(super) fn modal_transient_closure(
    completion: &crate::CompletedStructuralModalTransientStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let modal = modal_closure(
        &crate::CompletedStructuralModalStudy {
            result: completion.modal.clone(),
        },
        access,
    )?;
    let ExactStructuralStudyAnalysisClosure::StructuralModal {
        result_root: modal_result_root,
        receipt_root: modal_receipt_root,
        objects: modal_objects,
    } = modal
    else {
        return Err(protocol("modal closure has an invalid kind"));
    };
    let basis_state_root = completion
        .basis_state
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let transient_result_root = completion
        .transient
        .result
        .result_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let transient_receipt_root = completion
        .transient
        .result
        .receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let objects = unique_references(
        modal_objects
            .into_iter()
            .chain(
                completion
                    .basis_state
                    .objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(
                completion
                    .transient
                    .result
                    .result_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(
                completion
                    .transient
                    .result
                    .receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            ),
    )?;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralModalTransient {
            modal_result_root,
            modal_receipt_root,
            basis_state_root: Box::new(basis_state_root),
            transient_result_root: Box::new(transient_result_root),
            transient_receipt_root: Box::new(transient_receipt_root),
            objects,
        },
    )
}

pub(super) fn modal_response_spectrum_closure(
    completion: &crate::CompletedStructuralModalResponseSpectrumStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
            &completion.basis_state.objects,
            &completion.response.result_objects,
            &completion.response.receipt_objects,
            &completion.accuracy.objects,
            &completion.publication_baseline_usage.objects,
            &completion.publication.objects,
            &completion.resource_usage.objects,
            &completion.response.field_recovery_objects,
            &completion.envelope_state.objects,
        ],
        access,
    )?;
    let [modal_result_root, modal_receipt_root, basis_state_root, response_result_root, response_receipt_root, response_accuracy_root, response_publication_baseline_usage_root, response_publication_root, response_resource_usage_root, field_recovery_root, envelope_state_root] =
        roots;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralModalResponseSpectrum {
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            basis_state_root: Box::new(basis_state_root),
            response_result_root: Box::new(response_result_root),
            response_receipt_root: Box::new(response_receipt_root),
            response_accuracy_root: Box::new(response_accuracy_root),
            response_publication_baseline_usage_root: Box::new(
                response_publication_baseline_usage_root,
            ),
            response_resource_usage_root: Box::new(response_resource_usage_root),
            response_publication_root: Box::new(response_publication_root),
            field_recovery_root: Box::new(field_recovery_root),
            envelope_state_root: Box::new(envelope_state_root),
            objects,
        },
    )
}

pub(super) fn random_response_closure(
    completion: &crate::CompletedStructuralRandomResponseStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
            &completion.basis_state.objects,
            &completion.response.result_objects,
            &completion.response.receipt_objects,
            &completion.accuracy.objects,
            &completion.publication_baseline_usage.objects,
            &completion.publication.objects,
            &completion.resource_usage.objects,
            &completion.statistics_state.objects,
        ],
        access,
    )?;
    let [modal_result_root, modal_receipt_root, basis_state_root, response_result_root, response_receipt_root, response_accuracy_root, response_publication_baseline_usage_root, response_publication_root, response_resource_usage_root, statistics_state_root] =
        roots;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralRandomResponse {
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            basis_state_root: Box::new(basis_state_root),
            response_result_root: Box::new(response_result_root),
            response_receipt_root: Box::new(response_receipt_root),
            response_accuracy_root: Box::new(response_accuracy_root),
            response_publication_baseline_usage_root: Box::new(
                response_publication_baseline_usage_root,
            ),
            response_resource_usage_root: Box::new(response_resource_usage_root),
            response_publication_root: Box::new(response_publication_root),
            statistics_state_root: Box::new(statistics_state_root),
            objects,
        },
    )
}
