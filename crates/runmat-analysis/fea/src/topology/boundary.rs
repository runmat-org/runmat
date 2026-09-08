use super::{
    contract::{
        AnalysisProjectionError, BoundaryNormalOrientation, ProjectedBoundaryFace,
        ProjectedElement, ProjectedNode,
    },
    element::{cross, dot, norm, subtract},
};
use runmat_meshing_core::SolverBoundaryFace;

pub(super) fn project_boundary_face(
    face: &SolverBoundaryFace,
    node_indices: Vec<usize>,
    adjacent_element_indices: Vec<usize>,
    nodes: &[ProjectedNode],
    elements: &[ProjectedElement],
) -> Result<ProjectedBoundaryFace, AnalysisProjectionError> {
    let corners =
        node_indices
            .get(..3)
            .ok_or(AnalysisProjectionError::InvalidBoundaryGeometry {
                face_id: face.face_id,
            })?;
    let coordinates = [
        nodes[corners[0]].coordinates_m,
        nodes[corners[1]].coordinates_m,
        nodes[corners[2]].coordinates_m,
    ];
    let centroid =
        std::array::from_fn(|axis| coordinates.iter().map(|point| point[axis]).sum::<f64>() / 3.0);
    let mut area_vector = cross(
        subtract(coordinates[1], coordinates[0]),
        subtract(coordinates[2], coordinates[0]),
    )
    .map(|value| 0.5 * value);
    let area = norm(area_vector);
    if !area.is_finite() || area <= 0.0 || adjacent_element_indices.is_empty() {
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
