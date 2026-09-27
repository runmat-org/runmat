//! Preserve the imported thermal prefix when converting the remote structural suffix.
use super::*;

pub(super) fn study_result(
    result: &runmat_execution_runner_native::ExactStructuralStudyResult,
    source: DownloadedResultCache,
) -> Result<StructuralStudyNumericalResult> {
    let structural = structural_result(result, source.clone())?;
    let Some(thermal) = &result.thermal else {
        return Ok(structural);
    };
    let (thermal, thermal_source) =
        runmat_execution_runner_native::thermal_study_completion_from_source(
            thermal.as_ref().clone(),
            source,
        )
        .map_err(anyhow::Error::msg)?;
    Ok(StructuralStudyNumericalResult::ThermalPrestress {
        thermal: Box::new(thermal),
        thermal_source,
        structural: Box::new(structural),
    })
}
