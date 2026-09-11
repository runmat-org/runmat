use runmat_meshing_core::ElementOrder;

use super::{quadrature::volume_shape, AnalysisProjection, ProjectedElement, VolumePointLocation};

const MAXIMUM_NEWTON_ITERATIONS: usize = 32;
const REFERENCE_TOLERANCE: f64 = 1.0e-10;

impl AnalysisProjection {
    /// Locate a physical point in the canonical solid topology and derive the exact interpolation
    /// weights used by its element formulation. Points on a shared face use the lowest stable
    /// element identity, making ownership independent of traversal and partition layout.
    pub fn locate_volume_point(&self, position_m: [f64; 3]) -> Option<VolumePointLocation> {
        if position_m.iter().any(|coordinate| !coordinate.is_finite()) {
            return None;
        }
        let mut matches = self
            .elements
            .iter()
            .enumerate()
            .filter_map(|(element_index, element)| {
                let shape_values = locate_in_element(self, element, position_m)?;
                let node_identities = element
                    .node_indices
                    .iter()
                    .map(|index| self.nodes.get(*index).map(|node| node.stable_identity))
                    .collect::<Option<Vec<_>>>()?;
                Some(VolumePointLocation {
                    element_index,
                    element_identity: element.stable_identity,
                    position_m,
                    node_indices: element.node_indices.clone(),
                    node_identities,
                    shape_values,
                })
            })
            .collect::<Vec<_>>();
        matches.sort_by_key(|location| location.element_identity);
        matches.into_iter().next()
    }
}

fn locate_in_element(
    projection: &AnalysisProjection,
    element: &ProjectedElement,
    position_m: [f64; 3],
) -> Option<Vec<f64>> {
    let expected = match element.order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    };
    if element.node_indices.len() != expected {
        return None;
    }
    let coordinates = element
        .node_indices
        .iter()
        .map(|index| projection.nodes.get(*index).map(|node| node.coordinates_m))
        .collect::<Option<Vec<_>>>()?;
    let coordinate_scale = coordinates
        .iter()
        .flatten()
        .chain(position_m.iter())
        .fold(0.0_f64, |scale, coordinate| scale.max(coordinate.abs()));
    let physical_tolerance = (element.characteristic_length_m.abs() * 1.0e-10
        + coordinate_scale * f64::EPSILON * 64.0)
        .max(f64::MIN_POSITIVE);
    for axis in 0..3 {
        let (minimum, maximum) = coordinates.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), point| (minimum.min(point[axis]), maximum.max(point[axis])),
        );
        if position_m[axis] < minimum - physical_tolerance
            || position_m[axis] > maximum + physical_tolerance
        {
            return None;
        }
    }

    let corner_origin = coordinates[0];
    let corner_jacobian = std::array::from_fn(|physical_axis| {
        std::array::from_fn(|reference_axis| {
            coordinates[reference_axis + 1][physical_axis] - corner_origin[physical_axis]
        })
    });
    let corner_right = std::array::from_fn(|axis| position_m[axis] - corner_origin[axis]);
    let linear_start = solve_3x3(corner_jacobian, corner_right);
    let starts = [
        linear_start,
        Some([0.25, 0.25, 0.25]),
        Some([0.0, 0.0, 0.0]),
        Some([1.0, 0.0, 0.0]),
        Some([0.0, 1.0, 0.0]),
        Some([0.0, 0.0, 1.0]),
    ];
    starts.into_iter().flatten().find_map(|reference| {
        invert_isoparametric_point(
            element.order,
            &coordinates,
            position_m,
            physical_tolerance,
            reference,
        )
    })
}

fn invert_isoparametric_point(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    position_m: [f64; 3],
    physical_tolerance: f64,
    mut reference: [f64; 3],
) -> Option<Vec<f64>> {
    let expected = coordinates.len();
    for _ in 0..MAXIMUM_NEWTON_ITERATIONS {
        let barycentric = [
            1.0 - reference[0] - reference[1] - reference[2],
            reference[0],
            reference[1],
            reference[2],
        ];
        let (shape_values, derivatives) = volume_shape(order, barycentric);
        let mapped = weighted_position(coordinates, &shape_values[..expected]);
        let residual = std::array::from_fn(|axis| mapped[axis] - position_m[axis]);
        if norm(residual) <= physical_tolerance {
            if barycentric.iter().all(|coordinate| {
                *coordinate >= -REFERENCE_TOLERANCE && *coordinate <= 1.0 + REFERENCE_TOLERANCE
            }) {
                return Some(shape_values[..expected].to_vec());
            }
            return None;
        }
        let jacobian = std::array::from_fn(|physical_axis| {
            std::array::from_fn(|reference_axis| {
                coordinates
                    .iter()
                    .zip(&derivatives[..expected])
                    .map(|(node, derivative)| node[physical_axis] * derivative[reference_axis])
                    .sum()
            })
        });
        let correction = solve_3x3(jacobian, residual)?;
        let current_norm = norm(residual);
        let mut step = 1.0;
        let mut accepted = None;
        while step >= 1.0 / 256.0 {
            let candidate = std::array::from_fn(|axis| reference[axis] - step * correction[axis]);
            let candidate_barycentric = [
                1.0 - candidate[0] - candidate[1] - candidate[2],
                candidate[0],
                candidate[1],
                candidate[2],
            ];
            let (candidate_shape, _) = volume_shape(order, candidate_barycentric);
            let candidate_position = weighted_position(coordinates, &candidate_shape[..expected]);
            let candidate_residual =
                std::array::from_fn(|axis| candidate_position[axis] - position_m[axis]);
            if norm(candidate_residual) < current_norm {
                accepted = Some(candidate);
                break;
            }
            step *= 0.5;
        }
        reference = accepted?;
        if reference.iter().any(|coordinate| !coordinate.is_finite()) {
            return None;
        }
    }
    None
}

fn weighted_position(coordinates: &[[f64; 3]], shape_values: &[f64]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        coordinates
            .iter()
            .zip(shape_values)
            .map(|(coordinate, shape)| coordinate[axis] * shape)
            .sum()
    })
}

fn norm(value: [f64; 3]) -> f64 {
    value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

fn solve_3x3(matrix: [[f64; 3]; 3], right: [f64; 3]) -> Option<[f64; 3]> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    let scale = matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let singular_tolerance = scale.powi(3) * f64::EPSILON * 64.0;
    if !determinant.is_finite()
        || !scale.is_finite()
        || scale == 0.0
        || determinant.abs() <= singular_tolerance
    {
        return None;
    }
    let replace = |column: usize| {
        let mut replaced = matrix;
        for row in 0..3 {
            replaced[row][column] = right[row];
        }
        replaced[0][0] * (replaced[1][1] * replaced[2][2] - replaced[1][2] * replaced[2][1])
            - replaced[0][1] * (replaced[1][0] * replaced[2][2] - replaced[1][2] * replaced[2][0])
            + replaced[0][2] * (replaced[1][0] * replaced[2][1] - replaced[1][1] * replaced[2][0])
    };
    Some([
        replace(0) / determinant,
        replace(1) / determinant,
        replace(2) / determinant,
    ])
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    use super::{solve_3x3, volume_shape, AnalysisProjection};

    #[test]
    fn locates_linear_and_quadratic_tetrahedron_points_with_partition_neutral_weights() {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let projection =
                AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap();
            let location = projection
                .locate_volume_point([0.2, 0.3, 0.1])
                .expect("interior point should resolve");
            assert_eq!(location.element_index, 0);
            assert_eq!(
                location.node_indices.len(),
                match order {
                    ElementOrder::Tet4 => 4,
                    ElementOrder::Tet10 => 10,
                }
            );
            assert!((location.shape_values.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
            let recovered = std::array::from_fn::<_, 3, _>(|axis| {
                location
                    .node_indices
                    .iter()
                    .zip(&location.shape_values)
                    .map(|(node, weight)| projection.nodes[*node].coordinates_m[axis] * weight)
                    .sum::<f64>()
            });
            for (actual, expected) in recovered.into_iter().zip([0.2, 0.3, 0.1]) {
                assert!((actual - expected).abs() < 1.0e-11);
            }
        }
    }

    #[test]
    fn rejects_points_outside_the_solid_projection() {
        let projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
                .unwrap();
        assert!(projection.locate_volume_point([0.5, 0.5, 0.5]).is_none());
    }

    #[test]
    fn reference_solve_is_scale_aware_for_small_physical_elements() {
        let scale = 1.0e-8;
        let solution = solve_3x3(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
            [0.2 * scale, 0.3 * scale, 0.1 * scale],
        )
        .expect("well-conditioned small element Jacobian must remain invertible");
        for (actual, expected) in solution.into_iter().zip([0.2, 0.3, 0.1]) {
            assert!((actual - expected).abs() < 1.0e-14);
        }
    }

    #[test]
    fn quadratic_location_inverts_a_curved_isoparametric_map() {
        let mut projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet10))
                .unwrap();
        projection.nodes[4].coordinates_m[1] = 0.2;
        projection.nodes[5].coordinates_m[2] = 0.15;
        projection.nodes[7].coordinates_m[0] = 0.12;
        let expected_barycentric = [0.4, 0.2, 0.3, 0.1];
        let (expected_weights, _) = volume_shape(ElementOrder::Tet10, expected_barycentric);
        let position = std::array::from_fn(|axis| {
            projection.elements[0]
                .node_indices
                .iter()
                .zip(expected_weights)
                .map(|(node, weight)| projection.nodes[*node].coordinates_m[axis] * weight)
                .sum::<f64>()
        });
        let location = projection
            .locate_volume_point(position)
            .expect("curved Tet10 point should resolve through the isoparametric map");
        for (actual, expected) in location.shape_values.iter().zip(expected_weights) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}
