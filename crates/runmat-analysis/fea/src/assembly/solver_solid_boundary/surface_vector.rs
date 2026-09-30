//! One shape-function surface integration authority for intensities and total face forces.
use super::*;

#[derive(Clone, Copy)]
enum SurfaceVector {
    /// A resultant's frame is resolved once at the physical surface centroid.
    GlobalDensity([f64; 3]),
    /// An authored traction field's frame is evaluated at each physical quadrature point.
    AuthoredDensity([f64; 3]),
}

pub(super) fn apply_projected_force(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    target: &runmat_analysis_core::RegionTarget,
    force_components_n: [f64; 3],
) -> Result<SpatialProjectionEvidence, SolverMeshRegionMappingError> {
    let faces = boundary_faces_matching(projection, target);
    let area = faces.iter().map(|face| face.area_m2).sum::<f64>();
    if !area.is_finite() || area <= 0.0 {
        return Err(SolverMeshRegionMappingError::InvalidAnalysisProjection(
            format!(
                "excitation `{}` total face force requires finite positive surface area",
                load.excitation_id.0,
            ),
        ));
    }
    let centroid = projected_boundary_centroid(&faces);
    let force = resolve_vector(load, force_components_n, centroid, boundary_normal(&faces))?;
    // fi = (F / A) integral(Ni dA): changing triangulation density cannot change the
    // physical distribution. In particular planar Tri6 corners have zero consistent weight.
    integrate_surface_vector(
        projection,
        layout,
        rhs,
        load,
        &faces,
        SurfaceVector::GlobalDensity(force.map(|component| component / area)),
    )
}

pub(super) fn apply_projected_traction(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    target: &runmat_analysis_core::RegionTarget,
    traction_components_pa: [f64; 3],
) -> Result<SpatialProjectionEvidence, SolverMeshRegionMappingError> {
    integrate_surface_vector(
        projection,
        layout,
        rhs,
        load,
        &boundary_faces_matching(projection, target),
        SurfaceVector::AuthoredDensity(traction_components_pa),
    )
}

fn integrate_surface_vector(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    faces: &[&crate::topology::ProjectedBoundaryFace],
    vector: SurfaceVector,
) -> Result<SpatialProjectionEvidence, SolverMeshRegionMappingError> {
    let reference_point_m = projected_boundary_centroid(faces);
    let mut evidence = SpatialProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::BoundaryDunavantSeven,
        faces.len(),
        reference_point_m,
    );
    for face in faces {
        let mut frame_error = None;
        projection
            .visit_boundary_quadrature(face, |point| {
                let traction = match vector {
                    SurfaceVector::GlobalDensity(value) => value,
                    SurfaceVector::AuthoredDensity(value) => {
                        let normal = unit_vector(point.area_vector_weight_m2).ok();
                        let Ok(value) =
                            transform_vector(&load.policy.frame, value, point.position_m, normal)
                        else {
                            frame_error =
                                Some("coordinate frame is undefined on the selected boundary");
                            return;
                        };
                        value
                    }
                };
                let integrated = scale_vector(traction, point.area_weight_m2);
                evidence.add_requested_force(point.position_m, integrated);
                for (&node_index, &shape) in face.node_indices.iter().zip(point.shape_values()) {
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
                format!("excitation `{}` {error}", load.excitation_id.0,),
            ));
        }
    }
    Ok(evidence.finish())
}
