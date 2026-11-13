use runmat_meshing_core::ElementOrder;

use super::{quadrature::volume_shape, AnalysisProjection, ProjectedElement, VolumePointLocation};

const MAXIMUM_NEWTON_ITERATIONS: usize = 32;
const REFERENCE_TOLERANCE: f64 = 1.0e-10;

impl AnalysisProjection {
    /// Locate a physical point in the canonical solid topology and derive the exact interpolation
    /// weights used by its element formulation. Points on a shared face use the lowest stable
    /// element identity, making ownership independent of traversal and partition layout.
    pub fn locate_volume_point(
        &self,
        position_m: [f64; 3],
    ) -> Result<Option<VolumePointLocation>, super::AnalysisProjectionError> {
        use super::storage::{check_cancelled, reserve};
        if position_m.iter().any(|coordinate| !coordinate.is_finite()) {
            return Ok(None);
        }
        let mut owner = None;
        for (element_index, element) in self.elements.iter().enumerate() {
            check_cancelled()?;
            // Ownership is by stable identity. Equal identities preserve original first-hit order.
            if owner
                .as_ref()
                .is_some_and(|(_, identity, _)| *identity <= element.stable_identity)
            {
                continue;
            }
            let Some(shape_values) = locate_in_element(self, element, position_m)? else {
                continue;
            };
            if element
                .node_indices
                .iter()
                .any(|index| self.nodes.get(*index).is_none())
            {
                continue;
            }
            owner = Some((element_index, element.stable_identity, shape_values));
        }
        let Some((element_index, element_identity, shape_values)) = owner else {
            return Ok(None);
        };
        let element = &self.elements[element_index];
        let count = element.node_indices.len();
        let mut node_indices = reserve(count)?;
        node_indices.extend_from_slice(&element.node_indices);
        let mut node_identities = reserve(count)?;
        node_identities.extend(
            element
                .node_indices
                .iter()
                .map(|index| self.nodes[*index].stable_identity),
        );
        let mut weights = reserve(count)?;
        weights.extend_from_slice(&shape_values[..count]);
        Ok(Some(VolumePointLocation {
            element_index,
            element_identity,
            position_m,
            node_indices,
            node_identities,
            shape_values: weights,
        }))
    }
    /// Requested output heap; trial geometry and ownership use fixed stack storage.
    pub fn volume_point_location_resident_bytes(
        &self,
    ) -> Result<u64, super::AnalysisProjectionError> {
        use super::storage::{add, slots};
        let count = self
            .elements
            .iter()
            .map(|element| element.node_indices.len())
            .max()
            .unwrap_or(0);
        Ok(add(
            add(
                slots::<usize>(count)?,
                slots::<runmat_meshing_core::StableDigest>(count)?,
            )?,
            slots::<f64>(count)?,
        )?)
    }
}

fn locate_in_element(
    projection: &AnalysisProjection,
    element: &ProjectedElement,
    position_m: [f64; 3],
) -> Result<Option<[f64; 10]>, super::AnalysisProjectionError> {
    let expected = match element.order {
        ElementOrder::Tet4 => 4,
        ElementOrder::Tet10 => 10,
    };
    if element.node_indices.len() != expected {
        return Ok(None);
    }
    let mut coordinates = [[0.0; 3]; 10];
    for (coordinate, index) in coordinates.iter_mut().zip(&element.node_indices) {
        let Some(node) = projection.nodes.get(*index) else {
            return Ok(None);
        };
        *coordinate = node.coordinates_m.canonical_value();
    }
    let coordinates = &coordinates[..expected];
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
            return Ok(None);
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
    for reference in starts.into_iter().flatten() {
        if let Some(shape) = invert_isoparametric_point(
            element.order,
            coordinates,
            position_m,
            physical_tolerance,
            reference,
        )? {
            return Ok(Some(shape));
        }
    }
    Ok(None)
}

fn invert_isoparametric_point(
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    position_m: [f64; 3],
    physical_tolerance: f64,
    mut reference: [f64; 3],
) -> Result<Option<[f64; 10]>, super::AnalysisProjectionError> {
    let expected = coordinates.len();
    for _ in 0..MAXIMUM_NEWTON_ITERATIONS {
        let barycentric = [
            1.0 - reference[0] - reference[1] - reference[2],
            reference[0],
            reference[1],
            reference[2],
        ];
        let (shape_values, derivatives) = volume_shape(order, barycentric)?;
        let mapped = weighted_position(coordinates, &shape_values[..expected]);
        let residual = std::array::from_fn(|axis| mapped[axis] - position_m[axis]);
        if norm(residual) <= physical_tolerance {
            if barycentric.iter().all(|coordinate| {
                *coordinate >= -REFERENCE_TOLERANCE && *coordinate <= 1.0 + REFERENCE_TOLERANCE
            }) {
                return Ok(Some(shape_values));
            }
            return Ok(None);
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
        let Some(correction) = solve_3x3(jacobian, residual) else {
            return Ok(None);
        };
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
            let (candidate_shape, _) = volume_shape(order, candidate_barycentric)?;
            let candidate_position = weighted_position(coordinates, &candidate_shape[..expected]);
            let candidate_residual =
                std::array::from_fn(|axis| candidate_position[axis] - position_m[axis]);
            if norm(candidate_residual) < current_norm {
                accepted = Some(candidate);
                break;
            }
            step *= 0.5;
        }
        let Some(next) = accepted else {
            return Ok(None);
        };
        reference = next;
        if reference.iter().any(|coordinate| !coordinate.is_finite()) {
            return Ok(None);
        }
    }
    Ok(None)
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
mod tests;
