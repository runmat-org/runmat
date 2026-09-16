use std::collections::BTreeMap;

use runmat_analysis_core::ProcedureStepId;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralModalWorkload, StructuralStepWorkload,
};
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_meshing_core::SolverMeshArtifact;

use super::result_import::{
    import_numerical_analysis_objects, validate_complete_numerical_inventory,
    validate_numerical_closure_inventory,
};
use super::{
    protocol, ExactStructuralModalTransientAnalysisResult, ExactStructuralStudyAnalysisResult,
};
use crate::NativeExecutionResult;

pub(super) struct ModalResultImport<'a, Source> {
    pub source: &'a Source,
    pub workload: StructuralModalWorkload,
    pub result_root: ValueRef,
    pub receipt_root: ValueRef,
    pub objects: Vec<ValueRef>,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_modal_result<Source: CacheImport>(
    input: ModalResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let result_objects = import_numerical_analysis_objects(
        input.source,
        &input.result_root,
        &input.objects,
        AnalysisNumericalObjectKind::SolverResult,
        input.access,
        input.limits,
    )?;
    let receipt_objects = import_numerical_analysis_objects(
        input.source,
        &input.receipt_root,
        &input.objects,
        AnalysisNumericalObjectKind::NumericalReceipt,
        input.access,
        input.limits,
    )?;
    validate_numerical_closure_inventory(
        &result_objects,
        &receipt_objects,
        &input.objects,
        input.access,
    )?;
    let result =
        runmat_analysis_execution::decode_prepared_structural_modal_result(&result_objects)
            .map_err(protocol)?;
    result.validate_for(&input.workload).map_err(protocol)?;
    let numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_modal_receipt(&receipt_objects)
            .map_err(protocol)?;
    validate_modal_pair(&result_objects, &result, &numerical_receipt)?;
    Ok(ExactStructuralStudyAnalysisResult::StructuralModal {
        resolved_workload: Box::new(input.workload),
        result: Box::new(result),
        numerical_receipt,
    })
}

pub(super) struct ModalTransientResultImport<'a, Source> {
    pub source: &'a Source,
    pub workloads: BTreeMap<ProcedureStepId, StructuralStepWorkload>,
    pub modal_result_root: ValueRef,
    pub modal_receipt_root: ValueRef,
    pub basis_state_root: ValueRef,
    pub transient_result_root: ValueRef,
    pub transient_receipt_root: ValueRef,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a SolverMeshArtifact,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_modal_transient_result<Source: CacheImport>(
    input: ModalTransientResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let authored = runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: input.workloads,
    };
    let step_ids = authored.step_ids().map_err(protocol)?;
    let [modal_step, transient_step] = step_ids.as_slice() else {
        return Err(protocol(
            "modal-transient result requires exactly two authored steps",
        ));
    };
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        &authored
    else {
        return Err(protocol(
            "modal-transient workload inventory has changed kind",
        ));
    };
    let StructuralStepWorkload::ModalExtraction {
        workload: modal_workload,
    } = &workloads[modal_step]
    else {
        return Err(protocol("modal-transient producer is not modal extraction"));
    };
    let StructuralStepWorkload::ModalTransient { template, .. } = &workloads[transient_step] else {
        return Err(protocol(
            "modal-transient consumer is not a response template",
        ));
    };

    let modal_result_objects = import_numerical_analysis_objects(
        input.source,
        &input.modal_result_root,
        &input.objects,
        AnalysisNumericalObjectKind::SolverResult,
        input.access,
        input.limits,
    )?;
    let modal_receipt_objects = import_numerical_analysis_objects(
        input.source,
        &input.modal_receipt_root,
        &input.objects,
        AnalysisNumericalObjectKind::NumericalReceipt,
        input.access,
        input.limits,
    )?;
    let basis_state_objects = import_numerical_analysis_objects(
        input.source,
        &input.basis_state_root,
        &input.objects,
        AnalysisNumericalObjectKind::SemanticState,
        input.access,
        input.limits,
    )?;
    let transient_result_objects = import_numerical_analysis_objects(
        input.source,
        &input.transient_result_root,
        &input.objects,
        AnalysisNumericalObjectKind::SolverResult,
        input.access,
        input.limits,
    )?;
    let transient_receipt_objects = import_numerical_analysis_objects(
        input.source,
        &input.transient_receipt_root,
        &input.objects,
        AnalysisNumericalObjectKind::NumericalReceipt,
        input.access,
        input.limits,
    )?;
    validate_complete_numerical_inventory(
        &[
            &modal_result_objects,
            &modal_receipt_objects,
            &basis_state_objects,
            &transient_result_objects,
            &transient_receipt_objects,
        ],
        &input.objects,
        input.access,
    )?;

    let modal_result =
        runmat_analysis_execution::decode_prepared_structural_modal_result(&modal_result_objects)
            .map_err(protocol)?;
    modal_result
        .validate_for(modal_workload)
        .map_err(protocol)?;
    let modal_numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_modal_receipt(&modal_receipt_objects)
            .map_err(protocol)?;
    validate_modal_pair(
        &modal_result_objects,
        &modal_result,
        &modal_numerical_receipt,
    )?;
    let basis_state =
        runmat_analysis_execution::decode_prepared_semantic_state(&basis_state_objects)
            .map_err(protocol)?;
    if basis_state.source_result != modal_numerical_receipt.source_result()
        || basis_state
            .payload
            .modal_basis()
            .map(|basis| basis.identity())
            != Some(modal_result.basis_identity())
    {
        return Err(protocol(
            "published modal basis differs from its exact producer result",
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
    let transient_workload = template.clone().bind(transition).map_err(protocol)?;
    let transient_result =
        runmat_analysis_execution::decode_prepared_structural_transient_solver_result(
            &transient_result_objects,
        )
        .map_err(protocol)?;
    let transient_numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_transient_numerical_receipt(
            &transient_receipt_objects,
        )
        .map_err(protocol)?;
    transient_numerical_receipt
        .validate_for_result(&transient_result)
        .map_err(protocol)?;
    let expected_scope =
        runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
            &transient_workload,
            input.mesh,
            Some(&basis_state),
        )
        .map_err(protocol)?;
    if transient_result.manifest().scope != expected_scope {
        return Err(protocol(
            "modal-transient result names a different resolved workload",
        ));
    }
    Ok(
        ExactStructuralStudyAnalysisResult::StructuralModalTransient(Box::new(
            ExactStructuralModalTransientAnalysisResult {
                modal_workload: modal_workload.clone(),
                transient_workload: Box::new(transient_workload),
                modal_result: Box::new(modal_result),
                modal_numerical_receipt,
                basis_state,
                transient_result: Box::new(transient_result),
                transient_numerical_receipt,
            },
        )),
    )
}

pub(super) fn validate_modal_pair(
    result_objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    result: &runmat_analysis_fea::execution::StructuralModalSolverResult,
    receipt: &runmat_analysis_fea::execution::StructuralModalNumericalReceipt,
) -> NativeExecutionResult<()> {
    if result_objects.manifest.logical_identity != receipt.source_result().digest() {
        return Err(protocol(
            "modal result object differs from its numerical receipt source",
        ));
    }
    receipt
        .validate_against(receipt.source_result(), result)
        .map_err(protocol)
}
