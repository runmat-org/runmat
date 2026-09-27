use std::collections::BTreeMap;

use runmat_analysis_core::ProcedureStepId;
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, StructuralStepWorkload};
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::result_import::{
    import_numerical_analysis_objects, validate_complete_numerical_inventory,
};
use super::{
    protocol, ExactStructuralPrestressedModalAnalysisResult, ExactStructuralStudyAnalysisResult,
};
use crate::NativeExecutionResult;

pub(super) struct PrestressedModalResultImport<'a, Source> {
    pub source: &'a Source,
    pub workloads: BTreeMap<ProcedureStepId, StructuralStepWorkload>,
    pub preload_result_root: ValueRef,
    pub preload_receipt_root: ValueRef,
    pub prestress_state_root: ValueRef,
    pub modal_result_root: ValueRef,
    pub modal_receipt_root: ValueRef,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a SolverMeshArtifact,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_prestressed_modal_result<Source: CacheImport>(
    input: PrestressedModalResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let authored = runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: input.workloads,
    };
    let prefix = authored.prestress_prefix().map_err(protocol)?;
    let mut step_ids = vec![prefix.preload.step_id.clone()];
    step_ids.extend(prefix.consumer_step_ids);
    let [preload_step, modal_step] = step_ids.as_slice() else {
        return Err(protocol(
            "prestressed-modal result requires exactly two authored steps",
        ));
    };
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        authored
    else {
        return Err(protocol(
            "prestressed-modal workload inventory changed kind",
        ));
    };
    let Some(StructuralStepWorkload::Linear { workload: preload }) = workloads.get(preload_step)
    else {
        return Err(protocol(
            "prestressed-modal producer is not a structural preload",
        ));
    };
    let Some(StructuralStepWorkload::PrestressedModal { template }) = workloads.get(modal_step)
    else {
        return Err(protocol(
            "prestressed-modal consumer is not an unresolved modal template",
        ));
    };
    let components = [
        (
            &input.preload_result_root,
            AnalysisNumericalObjectKind::SolverResult,
        ),
        (
            &input.preload_receipt_root,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ),
        (
            &input.prestress_state_root,
            AnalysisNumericalObjectKind::SemanticState,
        ),
        (
            &input.modal_result_root,
            AnalysisNumericalObjectKind::SolverResult,
        ),
        (
            &input.modal_receipt_root,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ),
    ]
    .into_iter()
    .map(|(root, kind)| {
        import_numerical_analysis_objects(
            input.source,
            root,
            &input.objects,
            kind,
            input.access,
            input.limits,
        )
    })
    .collect::<NativeExecutionResult<Vec<_>>>()?;
    validate_complete_numerical_inventory(
        &components.iter().collect::<Vec<_>>(),
        &input.objects,
        input.access,
    )?;
    let preload_result =
        runmat_analysis_fea::execution::StructuralLinearWorkloadResult::from_canonical_bytes(
            &components[0].payload(),
        )
        .map_err(protocol)?;
    let preload_receipt =
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
            &components[1].payload(),
        )
        .map_err(protocol)?;
    let preload_result_identity = preload_receipt.source_result();
    if components[0].manifest.logical_identity != preload_result_identity.digest() {
        return Err(protocol(
            "prestressed-modal preload result manifest has another identity",
        ));
    }
    preload_receipt
        .validate_against(preload_result_identity, &preload_result)
        .map_err(protocol)?;
    let prestress_state = runmat_analysis_execution::decode_prepared_semantic_state(&components[2])
        .map_err(protocol)?;
    if prestress_state.source_result != preload_result_identity {
        return Err(protocol(
            "prestressed-modal state names another preload result",
        ));
    }
    let transition = runmat_analysis_fea::execution::CanonicalStateTransition::bind_upstream(
        &template.model,
        &template.step_id,
        &prestress_state,
        runmat_analysis_core::StateUse::Prestress,
        runmat_execution::Digest::from_bytes(*input.mesh.canonical_digest.bytes()),
    )
    .map_err(protocol)?;
    let modal_workload = (**template).clone().bind(transition).map_err(protocol)?;
    let modal_result =
        runmat_analysis_execution::decode_prepared_structural_modal_result(&components[3])
            .map_err(protocol)?;
    modal_result
        .validate_for(&modal_workload)
        .map_err(protocol)?;
    let modal_numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_modal_receipt(&components[4])
            .map_err(protocol)?;
    modal_numerical_receipt
        .validate_against(modal_numerical_receipt.source_result(), &modal_result)
        .map_err(protocol)?;
    runmat_analysis_fea::recover_completed_structural_prestressed_modal_run(
        &modal_workload,
        input.mesh,
        &prestress_state,
        &modal_result,
        &modal_numerical_receipt,
    )
    .map_err(protocol)?;
    Ok(
        ExactStructuralStudyAnalysisResult::StructuralPrestressedModal(Box::new(
            ExactStructuralPrestressedModalAnalysisResult {
                preload_workload: preload.clone(),
                modal_workload: Box::new(modal_workload),
                preload_result: Box::new(preload_result),
                preload_result_identity,
                preload_numerical_receipt: preload_receipt,
                prestress_state,
                modal_result: Box::new(modal_result),
                modal_numerical_receipt,
            },
        )),
    )
}
