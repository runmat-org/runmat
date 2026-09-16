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
    let [modal_result_objects, modal_receipt_objects, basis_state_objects, response_result_objects, response_receipt_objects, envelope_state_objects] =
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
                envelope_state,
            },
        )),
    )
}
