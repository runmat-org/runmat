use super::*;

pub(super) fn validate_structured_meshing_input(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
) -> Result<(), MeshingError> {
    validate_volume_meshing_options(options)?;
    validate_boundary_regions(input)
}

pub(super) fn validate_volume_meshing_options(
    options: &VolumeMeshingOptions,
) -> Result<(), MeshingError> {
    if !matches!(options.kind, MeshKindRequest::Solid) {
        return Err(MeshingError::UnsupportedMeshKind(options.kind));
    }
    if !matches!(options.element, VolumeElementKind::Tetrahedron4) {
        return Err(MeshingError::UnsupportedElementKind(options.element));
    }
    if options.max_elements == 0 {
        return Err(MeshingError::InvalidElementBudget);
    }
    if let MeshTargetSize::LengthM(length_m) = options.target_size {
        if !length_m.is_finite() || length_m <= 0.0 {
            return Err(MeshingError::InvalidTargetSize);
        }
    }
    if let Some(min_size_m) = options.min_size_m {
        if !min_size_m.is_finite() || min_size_m <= 0.0 {
            return Err(MeshingError::InvalidTargetSize);
        }
    }
    if let Some(max_size_m) = options.max_size_m {
        if !max_size_m.is_finite() || max_size_m <= 0.0 {
            return Err(MeshingError::InvalidTargetSize);
        }
    }
    if let (Some(min_size_m), Some(max_size_m)) = (options.min_size_m, options.max_size_m) {
        if min_size_m > max_size_m {
            return Err(MeshingError::InvalidTargetSize);
        }
    }
    if let Some(growth_rate) = options.growth_rate {
        if !growth_rate.is_finite() || growth_rate < 1.0 {
            return Err(MeshingError::InvalidTargetSize);
        }
    }
    Ok(())
}

pub(super) fn validate_boundary_regions(input: &BoundaryMeshInput) -> Result<(), MeshingError> {
    if input.region_ids.is_empty() {
        return Err(MeshingError::EmptyBoundaryRegions);
    }
    Ok(())
}
