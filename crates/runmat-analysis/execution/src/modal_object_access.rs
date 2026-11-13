use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralModalNumericalReceipt, StructuralModalSolverResult,
    StructuralModalWorkload,
};
use runmat_execution::Digest;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ObjectNamespace;

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects,
};

pub fn prepare_structural_modal_workload_objects(
    workload: &StructuralModalWorkload,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let payload = workload.canonical_bytes()?;
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralModalWorkload,
        workload.identity().digest(),
        &payload,
        chunk_bytes,
        ObjectNamespace::InputValue,
        limits,
    )
}

pub fn decode_prepared_structural_modal_workload(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralModalWorkload> {
    require_kind(
        objects,
        AnalysisNumericalObjectKind::StructuralModalWorkload,
    )?;
    let workload = StructuralModalWorkload::from_canonical_bytes(&objects.payload())?;
    if objects.manifest.logical_identity != workload.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural modal workload manifest differs from its canonical identity",
        ));
    }
    Ok(workload)
}

pub fn decode_prepared_structural_modal_result(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralModalSolverResult> {
    require_kind(objects, AnalysisNumericalObjectKind::SolverResult)?;
    StructuralModalSolverResult::from_canonical_bytes(&objects.payload()).map_err(Into::into)
}

pub fn decode_prepared_structural_modal_receipt(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralModalNumericalReceipt> {
    require_kind(objects, AnalysisNumericalObjectKind::NumericalReceipt)?;
    StructuralModalNumericalReceipt::from_canonical_bytes(&objects.payload()).map_err(Into::into)
}

fn require_kind(
    objects: &PreparedAnalysisObjects,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<()> {
    if objects.manifest.kind != kind
        || objects.manifest.payload_digest != Digest::sha256(objects.payload())
    {
        return Err(AnalysisBridgeError::Identity(
            "prepared modal object differs from its manifest",
        ));
    }
    Ok(())
}
