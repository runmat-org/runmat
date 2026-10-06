//! Transient numerical closure assembly; no physics is recomputed.
use super::*;

pub(super) fn transient_closure(
    result_objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    receipt_objects: &runmat_analysis_execution::PreparedAnalysisObjects,
    access: &ArtifactAccess,
) -> NativeExecutionResult<ExactStructuralStudyAnalysisClosure> {
    let result_root = result_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let receipt_root = receipt_objects
        .root_reference(access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let objects = unique_references(
        result_objects
            .unique_references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?
            .into_iter()
            .chain(
                receipt_objects
                    .unique_references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)?,
            ),
    )?;
    Ok(ExactStructuralStudyAnalysisClosure::StructuralTransient {
        result_root: Box::new(result_root),
        receipt_root: Box::new(receipt_root),
        objects,
    })
}
