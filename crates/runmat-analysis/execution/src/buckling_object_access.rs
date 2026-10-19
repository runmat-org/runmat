use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralBucklingNumericalReceipt,
    StructuralBucklingSolverResult, StructuralBucklingWorkload,
};
use runmat_execution::Digest;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects,
};

pub fn prepare_structural_buckling_workload_objects(
    workload: &StructuralBucklingWorkload,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralBucklingWorkload,
        workload.identity().digest(),
        &workload.canonical_bytes()?,
        chunk_bytes,
        ObjectNamespace::InputValue,
        limits,
    )
}

pub fn decode_prepared_structural_buckling_workload(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralBucklingWorkload> {
    require(
        objects,
        AnalysisNumericalObjectKind::StructuralBucklingWorkload,
    )?;
    let workload = StructuralBucklingWorkload::from_canonical_bytes(&objects.payload())?;
    if objects.manifest.logical_identity != workload.identity().digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural buckling workload manifest differs from its canonical identity",
        ));
    }
    Ok(workload)
}

pub fn decode_prepared_structural_buckling_result(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralBucklingSolverResult> {
    require(objects, AnalysisNumericalObjectKind::SolverResult)?;
    StructuralBucklingSolverResult::from_canonical_bytes(&objects.payload()).map_err(Into::into)
}

pub fn decode_prepared_structural_buckling_receipt(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralBucklingNumericalReceipt> {
    require(objects, AnalysisNumericalObjectKind::NumericalReceipt)?;
    StructuralBucklingNumericalReceipt::from_canonical_bytes(&objects.payload()).map_err(Into::into)
}

fn require(
    objects: &PreparedAnalysisObjects,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<()> {
    if objects.manifest.kind != kind
        || objects.manifest.payload_digest != Digest::sha256(objects.payload())
    {
        return Err(AnalysisBridgeError::Identity(
            "prepared buckling object differs from its manifest",
        ));
    }
    Ok(())
}
