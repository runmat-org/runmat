use super::{
    contract::{
        AnalysisProjectionError, BoundaryNormalOrientation, ProjectedBoundaryFace,
        ProjectedElement, ProjectedNode,
    },
    element::{dot, norm, subtract},
    quadrature::visit_boundary_quadrature,
};
use runmat_meshing_core::SolverBoundaryFace;

pub(super) fn project_boundary_face(
    face: &SolverBoundaryFace,
    node_indices: Vec<usize>,
    adjacent_element_indices: Vec<usize>,
    nodes: &[ProjectedNode],
    elements: &[ProjectedElement],
) -> Result<ProjectedBoundaryFace, AnalysisProjectionError> {
    let coordinates = node_indices
        .iter()
        .map(|index| nodes[*index].coordinates_m)
        .collect::<Vec<_>>();
    let mut area = 0.0;
    let mut first_moment_m3 = [0.0; 3];
    let mut area_vector = [0.0; 3];
    visit_boundary_quadrature(face.order, &coordinates, |point| {
        area += point.area_weight_m2;
        for axis in 0..3 {
            first_moment_m3[axis] += point.position_m[axis] * point.area_weight_m2;
            area_vector[axis] += point.area_vector_weight_m2[axis];
        }
    })
    .ok_or(AnalysisProjectionError::InvalidBoundaryGeometry {
        face_id: face.face_id,
    })?;
    let centroid = first_moment_m3.map(|value| value / area);
    if !area.is_finite()
        || area <= 0.0
        || norm(area_vector) <= f64::EPSILON
        || adjacent_element_indices.is_empty()
    {
        return Err(AnalysisProjectionError::InvalidBoundaryGeometry {
            face_id: face.face_id,
        });
    }
    let interior = elements[adjacent_element_indices[0]].centroid_m;
    if dot(area_vector, subtract(interior, centroid)) > 0.0 {
        area_vector = area_vector.map(|value| -value);
    }
    Ok(ProjectedBoundaryFace {
        face_id: face.face_id,
        stable_identity: face.stable_identity,
        order: face.order,
        role: face.role,
        node_indices,
        adjacent_element_indices,
        provenance: face.provenance.clone(),
        centroid_m: centroid,
        oriented_area_vector_m2: area_vector,
        normal_orientation: if face.role
            == runmat_meshing_core::BoundaryFaceRole::ConformalInterface
        {
            BoundaryNormalOrientation::FirstAdjacentElementOutward
        } else {
            BoundaryNormalOrientation::ExteriorOutward
        },
        area_m2: area,
    })
}
