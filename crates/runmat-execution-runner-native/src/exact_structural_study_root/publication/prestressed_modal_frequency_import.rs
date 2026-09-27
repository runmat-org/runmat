use std::collections::BTreeMap;

use runmat_analysis_core::{ProcedureStepId, StateUse};
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, StructuralStepWorkload};
use runmat_execution::{value::ValueRef, Digest};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::result_import::import_numerical_analysis_objects;
use super::{
    protocol, reference_inventory, unique_references, ExactStructuralFrequencyClosure,
    ExactStructuralPrestressedModalFrequencyAnalysisResult, ExactStructuralStudyAnalysisResult,
};
use crate::NativeExecutionResult;

pub(super) struct PrestressedModalFrequencyResultImport<'a, Source> {
    pub source: &'a Source,
    pub workloads: BTreeMap<ProcedureStepId, StructuralStepWorkload>,
    pub preload_result_root: ValueRef,
    pub preload_receipt_root: ValueRef,
    pub prestress_state_root: ValueRef,
    pub modal_result_root: ValueRef,
    pub modal_receipt_root: ValueRef,
    pub basis_state_root: ValueRef,
    pub frequency: ExactStructuralFrequencyClosure,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a SolverMeshArtifact,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_prestressed_modal_frequency_result<Source: CacheImport>(
    input: PrestressedModalFrequencyResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let authored = runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: input.workloads,
    };
    let prefix = authored.prestress_prefix().map_err(protocol)?;
    let mut step_ids = vec![prefix.preload.step_id.clone()];
    step_ids.extend(prefix.consumer_step_ids);
    let [preload_step, modal_step, frequency_step] = step_ids.as_slice() else {
        return Err(protocol(
            "prestressed modal-frequency result requires exactly three authored steps",
        ));
    };
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        authored
    else {
        return Err(protocol(
            "prestressed modal-frequency workload inventory changed kind",
        ));
    };
    let Some(StructuralStepWorkload::Linear { workload: preload }) = workloads.get(preload_step)
    else {
        return Err(protocol(
            "prestressed modal-frequency producer is not a structural preload",
        ));
    };
    let Some(StructuralStepWorkload::PrestressedModal {
        template: modal_template,
    }) = workloads.get(modal_step)
    else {
        return Err(protocol(
            "prestressed modal-frequency basis producer is not a prestressed-modal template",
        ));
    };
    let Some(StructuralStepWorkload::PrestressedFrequency {
        template: frequency_template,
    }) = workloads.get(frequency_step)
    else {
        return Err(protocol(
            "prestressed modal-frequency consumer is not a harmonic template",
        ));
    };

    let prefix = [
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
        (
            &input.basis_state_root,
            AnalysisNumericalObjectKind::SemanticState,
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

    let preload_result =
        runmat_analysis_fea::execution::StructuralLinearWorkloadResult::from_canonical_bytes(
            &prefix[0].payload(),
        )
        .map_err(protocol)?;
    let preload_receipt = runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
        &prefix[1].payload(),
    )
    .map_err(protocol)?;
    let preload_result_identity = preload_receipt.source_result();
    if prefix[0].manifest.logical_identity != preload_result_identity.digest() {
        return Err(protocol(
            "prestressed modal-frequency preload result manifest has another identity",
        ));
    }
    preload_receipt
        .validate_against(preload_result_identity, &preload_result)
        .map_err(protocol)?;
    let prestress_state =
        runmat_analysis_execution::decode_prepared_semantic_state(&prefix[2]).map_err(protocol)?;
    if prestress_state.source_result != preload_result_identity {
        return Err(protocol(
            "prestressed modal-frequency state names another preload result",
        ));
    }

    let topology_digest = Digest::from_bytes(*input.mesh.canonical_digest.bytes());
    let modal_transition = runmat_analysis_execution::bind_upstream_state_transition(
        &modal_template.model,
        &modal_template.step_id,
        &prestress_state,
        StateUse::Prestress,
        topology_digest,
    )
    .map_err(protocol)?;
    let modal_workload = (**modal_template)
        .clone()
        .bind(modal_transition)
        .map_err(protocol)?;
    let modal_result =
        runmat_analysis_execution::decode_prepared_structural_modal_result(&prefix[3])
            .map_err(protocol)?;
    modal_result
        .validate_for(&modal_workload)
        .map_err(protocol)?;
    let modal_receipt =
        runmat_analysis_execution::decode_prepared_structural_modal_receipt(&prefix[4])
            .map_err(protocol)?;
    if prefix[3].manifest.logical_identity != modal_receipt.source_result().digest() {
        return Err(protocol(
            "prestressed modal-frequency modal result differs from its receipt source",
        ));
    }
    modal_receipt
        .validate_against(modal_receipt.source_result(), &modal_result)
        .map_err(protocol)?;
    runmat_analysis_fea::recover_completed_structural_prestressed_modal_run(
        &modal_workload,
        input.mesh,
        &prestress_state,
        &modal_result,
        &modal_receipt,
    )
    .map_err(protocol)?;

    let basis_state =
        runmat_analysis_execution::decode_prepared_semantic_state(&prefix[5]).map_err(protocol)?;
    if basis_state.source_result != modal_receipt.source_result()
        || basis_state
            .payload
            .modal_basis()
            .map(|basis| basis.identity())
            != Some(modal_result.basis_identity())
    {
        return Err(protocol(
            "published prestressed modal basis differs from its exact producer result",
        ));
    }
    let prestress_transition = runmat_analysis_execution::bind_upstream_state_transition(
        &frequency_template.model,
        &frequency_template.step_id,
        &prestress_state,
        StateUse::Prestress,
        topology_digest,
    )
    .map_err(protocol)?;
    runmat_analysis_execution::bind_upstream_state_transition(
        &frequency_template.model,
        &frequency_template.step_id,
        &basis_state,
        StateUse::ModalBasis,
        topology_digest,
    )
    .map_err(protocol)?;
    let frequency_workload = (**frequency_template)
        .clone()
        .bind(prestress_transition, Some(basis_state.identity))
        .map_err(protocol)?;
    let frequency = super::frequency_closure_import::import_frequency_closure(
        super::frequency_closure_import::FrequencyClosureImport {
            source: input.source,
            workload: frequency_workload,
            closure: input.frequency,
            objects: input.objects.clone(),
            mesh: input.mesh,
            prestress_state: Some(&prestress_state),
            modal_state: Some(&basis_state),
            access: input.access,
            limits: input.limits,
        },
    )?;

    let mut admitted = prefix
        .iter()
        .map(|component| {
            component
                .unique_references(
                    input.access,
                    runmat_execution::value::ValueRefKind::ResultObject,
                )
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    admitted.extend(frequency.object_catalog.objects.iter().cloned());
    if reference_inventory(&unique_references(admitted)?)? != reference_inventory(&input.objects)? {
        return Err(protocol(
            "prestressed modal-frequency closure contains missing or unreferenced objects",
        ));
    }

    Ok(
        ExactStructuralStudyAnalysisResult::StructuralPrestressedModalFrequency(Box::new(
            ExactStructuralPrestressedModalFrequencyAnalysisResult {
                preload_workload: preload.clone(),
                modal_workload: Box::new(modal_workload),
                preload_result: Box::new(preload_result),
                preload_result_identity,
                preload_numerical_receipt: preload_receipt,
                prestress_state,
                modal_result: Box::new(modal_result),
                modal_numerical_receipt: modal_receipt,
                basis_state,
                frequency,
            },
        )),
    )
}
