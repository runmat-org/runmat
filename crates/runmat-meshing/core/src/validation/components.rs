use crate::contracts::AnalysisMeshArtifact;

use super::{volume_component_count, AnalysisMeshValidationError};

pub(super) fn validate_volume_component_count(
    mesh: &AnalysisMeshArtifact,
    max_component_count: Option<usize>,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(max_component_count) = max_component_count else {
        return Ok(());
    };
    let component_count = volume_component_count(mesh);
    if component_count > max_component_count {
        return Err(AnalysisMeshValidationError::VolumeComponentCountExceeded {
            component_count,
            max_component_count,
        });
    }
    Ok(())
}
