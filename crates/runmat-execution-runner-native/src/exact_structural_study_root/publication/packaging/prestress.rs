//! Prestress numerical closure assembly; no physics is recomputed.
use super::*;

pub(super) fn buckling_closure(
    completion: &crate::CompletedStructuralBucklingSequenceStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.preload.objects,
            &completion.preload.numerical_receipt_objects,
            &completion.prestress_state.objects,
            &completion.buckling.objects,
            &completion.buckling.numerical_receipt_objects,
        ],
        access,
    )?;
    let [preload_result_root, preload_receipt_root, prestress_state_root, buckling_result_root, buckling_receipt_root] =
        roots;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralBuckling {
        preload_result_root: Box::new(preload_result_root),
        preload_receipt_root: Box::new(preload_receipt_root),
        prestress_state_root: Box::new(prestress_state_root),
        buckling_result_root: Box::new(buckling_result_root),
        buckling_receipt_root: Box::new(buckling_receipt_root),
        objects,
    })
}

pub(super) fn prestressed_modal_closure(
    completion: &crate::CompletedStructuralPrestressedModalStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, objects) = numerical_closure_roots(
        [
            &completion.preload.objects,
            &completion.preload.numerical_receipt_objects,
            &completion.prestress_state.objects,
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
        ],
        access,
    )?;
    let [preload_result_root, preload_receipt_root, prestress_state_root, modal_result_root, modal_receipt_root] =
        roots;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralPrestressedModal {
            preload_result_root: Box::new(preload_result_root),
            preload_receipt_root: Box::new(preload_receipt_root),
            prestress_state_root: Box::new(prestress_state_root),
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            objects,
        },
    )
}

pub(super) fn prestressed_frequency_closure(
    completion: &crate::CompletedStructuralPrestressedFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let preload_result_root = completion
        .preload
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let preload_receipt_root = completion
        .preload
        .numerical_receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let prestress_state_root = completion
        .prestress_state
        .objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let (frequency, frequency_objects) = frequency_components(&completion.frequency, access)?;
    let objects = unique_references(
        completion
            .preload
            .objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?
            .into_iter()
            .chain(
                completion
                    .preload
                    .numerical_receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(
                completion
                    .prestress_state
                    .objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            )
            .chain(frequency_objects),
    )?;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralPrestressedFrequency {
            preload_result_root: Box::new(preload_result_root),
            preload_receipt_root: Box::new(preload_receipt_root),
            prestress_state_root: Box::new(prestress_state_root),
            frequency: Box::new(frequency),
            objects,
        },
    )
}

pub(super) fn prestressed_modal_frequency_closure(
    completion: &crate::CompletedStructuralPrestressedModalFrequencyStudy,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let (roots, prefix_objects) = numerical_closure_roots(
        [
            &completion.preload.objects,
            &completion.preload.numerical_receipt_objects,
            &completion.prestress_state.objects,
            &completion.modal.objects,
            &completion.modal.numerical_receipt_objects,
            &completion.basis_state.objects,
        ],
        access,
    )?;
    let [preload_result_root, preload_receipt_root, prestress_state_root, modal_result_root, modal_receipt_root, basis_state_root] =
        roots;
    let (frequency, frequency_objects) = frequency_components(&completion.frequency, access)?;
    let objects = unique_references(prefix_objects.into_iter().chain(frequency_objects))?;
    Ok(
        ExactStructuralStudyAnalysisClosure::StructuralPrestressedModalFrequency {
            preload_result_root: Box::new(preload_result_root),
            preload_receipt_root: Box::new(preload_receipt_root),
            prestress_state_root: Box::new(prestress_state_root),
            modal_result_root: Box::new(modal_result_root),
            modal_receipt_root: Box::new(modal_receipt_root),
            basis_state_root: Box::new(basis_state_root),
            frequency: Box::new(frequency),
            objects,
        },
    )
}
