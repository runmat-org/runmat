use runmat_meshing_core::{
    MeshKindRequest, MeshTargetSize, VolumeElementKind, VolumeMeshingOptions,
};

use super::SolidMeshingError;

pub(super) fn validate_solid_options(
    options: &VolumeMeshingOptions,
) -> Result<(), SolidMeshingError> {
    if !matches!(options.kind, MeshKindRequest::Solid) {
        return Err(SolidMeshingError::UnsupportedMeshKind(options.kind));
    }
    if !matches!(options.element, VolumeElementKind::Tetrahedron4) {
        return Err(SolidMeshingError::UnsupportedElementKind(options.element));
    }
    if options.max_elements == 0 {
        return Err(SolidMeshingError::InvalidElementBudget);
    }
    validate_target_size(options)
}

fn validate_target_size(options: &VolumeMeshingOptions) -> Result<(), SolidMeshingError> {
    if let MeshTargetSize::LengthM(length) = options.target_size {
        if !length.is_finite() || length <= 0.0 {
            return Err(SolidMeshingError::InvalidTargetSize);
        }
    }
    if let (Some(min), Some(max)) = (options.min_size_m, options.max_size_m) {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || max <= 0.0 || min > max {
            return Err(SolidMeshingError::InvalidTargetSize);
        }
    }
    if let Some(growth_rate) = options.growth_rate {
        if !growth_rate.is_finite() || growth_rate < 1.0 {
            return Err(SolidMeshingError::InvalidTargetSize);
        }
    }
    Ok(())
}
