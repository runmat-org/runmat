use std::collections::BTreeMap;

use runmat_analysis_core::ProcedureStepId;
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, StructuralStepWorkload};
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::result_import::import_numerical_analysis_objects;
use super::{
    protocol, reference_inventory, unique_references, ExactStructuralFrequencyClosure,
    ExactStructuralPrestressedFrequencyAnalysisResult, ExactStructuralStudyAnalysisResult,
};
use crate::NativeExecutionResult;

pub(super) struct PrestressedFrequencyResultImport<'a, Source> {
    pub source: &'a Source,
    pub workloads: BTreeMap<ProcedureStepId, StructuralStepWorkload>,
    pub preload_result_root: ValueRef,
    pub preload_receipt_root: ValueRef,
    pub prestress_state_root: ValueRef,
    pub frequency: ExactStructuralFrequencyClosure,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a SolverMeshArtifact,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_prestressed_frequency_result<Source: CacheImport>(
    input: PrestressedFrequencyResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let authored = runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: input.workloads,
    };
    let prefix = authored.prestress_prefix().map_err(protocol)?;
    let mut step_ids = vec![prefix.preload.step_id.clone()];
    step_ids.extend(prefix.consumer_step_ids);
    let [preload_step, frequency_step] = step_ids.as_slice() else {
        return Err(protocol(
            "prestressed-frequency result requires exactly two authored steps",
        ));
    };
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        authored
    else {
        return Err(protocol(
            "prestressed-frequency workload inventory changed kind",
        ));
    };
    let Some(StructuralStepWorkload::Linear { workload: preload }) = workloads.get(preload_step)
    else {
        return Err(protocol(
            "prestressed-frequency producer is not a structural preload",
        ));
    };
    let Some(StructuralStepWorkload::PrestressedFrequency { template }) =
        workloads.get(frequency_step)
    else {
        return Err(protocol(
            "prestressed-frequency consumer is not an unresolved frequency template",
        ));
    };
    let preload_components = [
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
            &preload_components[0].payload(),
        )
        .map_err(protocol)?;
    let preload_receipt =
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
            &preload_components[1].payload(),
        )
        .map_err(protocol)?;
    let preload_result_identity = preload_receipt.source_result();
    if preload_components[0].manifest.logical_identity != preload_result_identity.digest() {
        return Err(protocol(
            "prestressed-frequency preload result manifest has another identity",
        ));
    }
    preload_receipt
        .validate_against(preload_result_identity, &preload_result)
        .map_err(protocol)?;
    let prestress_state =
        runmat_analysis_execution::decode_prepared_semantic_state(&preload_components[2])
            .map_err(protocol)?;
    if prestress_state.source_result != preload_result_identity {
        return Err(protocol(
            "prestressed-frequency state names another preload result",
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
    let frequency_workload = (**template)
        .clone()
        .bind(transition, None)
        .map_err(protocol)?;
    let frequency = super::frequency_closure_import::import_frequency_closure(
        super::frequency_closure_import::FrequencyClosureImport {
            source: input.source,
            workload: frequency_workload,
            closure: input.frequency,
            objects: input.objects.clone(),
            mesh: input.mesh,
            prestress_state: Some(&prestress_state),
            modal_state: None,
            access: input.access,
            limits: input.limits,
        },
    )?;
    let mut admitted = preload_components
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
            "prestressed-frequency closure contains missing or unreferenced objects",
        ));
    }
    Ok(
        ExactStructuralStudyAnalysisResult::StructuralPrestressedFrequency(Box::new(
            ExactStructuralPrestressedFrequencyAnalysisResult {
                preload_workload: preload.clone(),
                preload_result: Box::new(preload_result),
                preload_result_identity,
                preload_numerical_receipt: preload_receipt,
                prestress_state,
                frequency,
            },
        )),
    )
}
