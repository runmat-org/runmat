use std::collections::BTreeSet;

use crate::{artifact::AnalysisMeshArtifact, topology::VolumeElementKind};

use super::{
    geometry::{element_tetrahedron_points, tetrahedron_volume_m3},
    AnalysisMeshValidationError,
};

pub(super) fn validate_required_boundary_regions(
    mesh: &AnalysisMeshArtifact,
    required_region_ids: &[String],
) -> Result<(), AnalysisMeshValidationError> {
    if required_region_ids.is_empty() {
        return Ok(());
    }
    let present = mesh
        .boundary_faces
        .iter()
        .flat_map(|face| face.region_ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let recovered = mesh
        .boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .flat_map(|face| face.region_ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    for region_id in required_region_ids {
        if !present.contains(region_id.as_str()) {
            return Err(AnalysisMeshValidationError::MissingRequiredBoundaryRegion {
                region_id: region_id.clone(),
            });
        }
        if !recovered.contains(region_id.as_str()) {
            return Err(
                AnalysisMeshValidationError::MissingRequiredBoundaryRegionRecovery {
                    region_id: region_id.clone(),
                },
            );
        }
    }
    Ok(())
}

pub(super) fn validate_required_material_regions(
    mesh: &AnalysisMeshArtifact,
    required_region_ids: &[String],
) -> Result<(), AnalysisMeshValidationError> {
    if required_region_ids.is_empty() {
        return Ok(());
    }
    let present = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    let positive_volume = mesh
        .volume_elements
        .iter()
        .filter(|element| {
            element.kind == VolumeElementKind::Tetrahedron4 && element.node_ids.len() == 4
        })
        .filter(|element| {
            let Some(points) = element_tetrahedron_points(mesh, element.node_ids.as_slice()) else {
                return false;
            };
            let volume_m3 = tetrahedron_volume_m3(points);
            volume_m3.is_finite() && volume_m3 > f64::EPSILON
        })
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    for region_id in required_region_ids {
        if !present.contains(region_id.as_str()) {
            return Err(AnalysisMeshValidationError::MissingRequiredMaterialRegion {
                region_id: region_id.clone(),
            });
        }
        if !positive_volume.contains(region_id.as_str()) {
            return Err(
                AnalysisMeshValidationError::MissingRequiredMaterialRegionCoverage {
                    region_id: region_id.clone(),
                },
            );
        }
    }
    Ok(())
}
