use super::*;
use crate::excitation::field::NodalVectorField;
use runmat_execution::Digest;

#[derive(Clone, Copy)]
pub(super) enum BodyDensity<'a> {
    Uniform([f64; 3]),
    Nodal(&'a NodalVectorField),
}

impl BodyDensity<'_> {
    fn validate_mesh(
        self,
        projection: &AnalysisProjection,
    ) -> Result<(), SolverMeshRegionMappingError> {
        if let Self::Nodal(field) = self {
            field
                .validate_mesh(
                    Digest::from_bytes(*projection.mesh_digest.bytes()),
                    projection.nodes.len(),
                )
                .map_err(|error| {
                    SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
                })?;
        }
        Ok(())
    }

    fn at(
        self,
        nodes: &[usize],
        shapes: &[f64],
    ) -> Result<[f64; 3], crate::excitation::RealizationError> {
        match self {
            Self::Uniform(value) => Ok(value),
            Self::Nodal(field) => field.real_at(nodes, shapes),
        }
    }
}

pub(super) fn apply_projected_body_force(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    density: BodyDensity<'_>,
) -> Result<(usize, Option<SpatialProjectionEvidence>), SolverMeshRegionMappingError> {
    density.validate_mesh(projection)?;
    let elements = projection
        .elements
        .iter()
        .filter(|element| match &load.target {
            RealizedExcitationTarget::WholeModel => true,
            RealizedExcitationTarget::Region(target) => {
                target.selects_volume_element(&element.region_id, &element.provenance)
            }
            _ => false,
        })
        .collect::<Vec<_>>();
    if elements.is_empty() {
        return Ok((0, None));
    }
    let reference_point_m = projected_volume_centroid(&elements);
    let mut evidence = SpatialProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::VolumeKeastFifteen,
        elements.len(),
        reference_point_m,
    );
    for element in &elements {
        let mut frame_error = None;
        projection
            .visit_element_quadrature(element, |point| {
                let density_components_n_per_m3 =
                    match density.at(&element.node_indices, point.shape_values()) {
                        Ok(value) => value,
                        Err(_) => {
                            frame_error =
                                Some("nodal field cannot be interpolated in the selected element");
                            return;
                        }
                    };
                let Ok(density_n_per_m3) = transform_vector(
                    &load.policy.frame,
                    density_components_n_per_m3,
                    point.position_m,
                    None,
                ) else {
                    frame_error = Some("coordinate frame is undefined in the selected volume");
                    return;
                };
                let integrated = scale_vector(density_n_per_m3, point.volume_weight_m3);
                evidence.add_requested_force(point.position_m, integrated);
                for (&node_index, &shape) in element.node_indices.iter().zip(point.shape_values()) {
                    let nodal_force = scale_vector(integrated, shape);
                    add_translational_rhs(layout, rhs, node_index, nodal_force);
                    evidence.add_projected_force(node_index, nodal_force);
                }
            })
            .map_err(|error| {
                SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
            })?;
        if let Some(error) = frame_error {
            return Err(SolverMeshRegionMappingError::InvalidAnalysisProjection(
                format!("excitation `{}` {error}", load.excitation_id.0),
            ));
        }
    }
    Ok((elements.len(), Some(evidence.finish())))
}

fn projected_volume_centroid(elements: &[&crate::topology::ProjectedElement]) -> [f64; 3] {
    let mut first_moment = [0.0; 3];
    let mut volume = 0.0;
    for element in elements {
        add_scaled(
            &mut first_moment,
            element.centroid_m,
            element.signed_volume_m3,
        );
        volume += element.signed_volume_m3;
    }
    scale_vector(first_moment, 1.0 / volume)
}
