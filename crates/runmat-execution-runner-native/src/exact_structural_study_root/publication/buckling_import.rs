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
use super::{protocol, ExactStructuralBucklingAnalysisResult, ExactStructuralStudyAnalysisResult};
use crate::NativeExecutionResult;

pub(super) struct BucklingResultImport<'a, Source> {
    pub source: &'a Source,
    pub workloads: BTreeMap<ProcedureStepId, StructuralStepWorkload>,
    pub preload_result_root: ValueRef,
    pub preload_receipt_root: ValueRef,
    pub prestress_state_root: ValueRef,
    pub buckling_result_root: ValueRef,
    pub buckling_receipt_root: ValueRef,
    pub objects: Vec<ValueRef>,
    pub mesh: &'a SolverMeshArtifact,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import_buckling_result<Source: CacheImport>(
    input: BucklingResultImport<'_, Source>,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisResult> {
    let authored = runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: input.workloads,
    };
    let prefix = authored.prestress_prefix().map_err(protocol)?;
    let mut step_ids = vec![prefix.preload.step_id.clone()];
    step_ids.extend(prefix.consumer_step_ids);
    let [preload_step, buckling_step] = step_ids.as_slice() else {
        return Err(protocol(
            "buckling result requires exactly two authored steps",
        ));
    };
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        authored
    else {
        return Err(protocol("buckling workload inventory changed kind"));
    };
    let Some(StructuralStepWorkload::Linear { workload: preload }) = workloads.get(preload_step)
    else {
        return Err(protocol("buckling producer is not a structural preload"));
    };
    let Some(StructuralStepWorkload::Buckling { template }) = workloads.get(buckling_step) else {
        return Err(protocol(
            "buckling consumer is not an unresolved buckling template",
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
            &input.buckling_result_root,
            AnalysisNumericalObjectKind::SolverResult,
        ),
        (
            &input.buckling_receipt_root,
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
            "buckling preload result manifest has another identity",
        ));
    }
    preload_receipt
        .validate_against(preload_result_identity, &preload_result)
        .map_err(protocol)?;
    let prestress_state = runmat_analysis_execution::decode_prepared_semantic_state(&components[2])
        .map_err(protocol)?;
    if prestress_state.source_result != preload_result_identity {
        return Err(protocol(
            "buckling prestress state names another preload result",
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
    let buckling_workload = (**template).clone().bind(transition).map_err(protocol)?;
    let buckling_result =
        runmat_analysis_execution::decode_prepared_structural_buckling_result(&components[3])
            .map_err(protocol)?;
    buckling_result
        .validate_for(&buckling_workload)
        .map_err(protocol)?;
    let buckling_numerical_receipt =
        runmat_analysis_execution::decode_prepared_structural_buckling_receipt(&components[4])
            .map_err(protocol)?;
    buckling_numerical_receipt
        .validate_against(buckling_numerical_receipt.source_result(), &buckling_result)
        .map_err(protocol)?;
    Ok(ExactStructuralStudyAnalysisResult::StructuralBuckling(
        Box::new(ExactStructuralBucklingAnalysisResult {
            preload_workload: preload.clone(),
            buckling_workload: Box::new(buckling_workload),
            preload_result: Box::new(preload_result),
            preload_result_identity,
            preload_numerical_receipt: preload_receipt,
            prestress_state,
            buckling_result: Box::new(buckling_result),
            buckling_numerical_receipt,
        }),
    ))
}
