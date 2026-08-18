use std::collections::BTreeMap;

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    MeshingCancellationSignal, SolverMeshTopology, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};

const BARYCENTRIC_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];
const REFERENCE_TETRAHEDRON: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub(super) fn validate(
    topology: &SolverMeshTopology,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    let node_by_id = topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut work = 0_u64;
    for element in &topology.volume_elements {
        let coordinates = element
            .node_ids
            .iter()
            .map(|node_id| {
                node_by_id
                    .get(node_id)
                    .copied()
                    .ok_or_else(|| invalid("Tet10 Jacobian references an absent node"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let coordinates: [[f64; 3]; 10] = coordinates
            .try_into()
            .map_err(|_| invalid("Tet10 Jacobian requires exactly ten nodes"))?;
        certify_element(
            coordinates,
            maximum_search_work,
            maximum_recursion_depth,
            cancellation_check_interval,
            cancellation,
            &mut work,
        )?;
    }
    Ok(())
}

fn certify_element(
    coordinates: [[f64; 3]; 10],
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    cancellation_check_interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
    work: &mut u64,
) -> Result<(), DelaunaySolverTopologyError> {
    let orientation = match orient3d([
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
    ])
    .map_err(|failure| invalid(format!("Tet10 corner orientation failed: {failure:?}")))?
    {
        PredicateSign::Positive => -1.0,
        PredicateSign::Negative => 1.0,
        PredicateSign::Zero => return Err(invalid("Tet10 corners are exactly coplanar")),
    };
    let mut pending = vec![(REFERENCE_TETRAHEDRON, 0_u32)];
    while let Some((cell, depth)) = pending.pop() {
        *work = work
            .checked_add(4)
            .ok_or_else(|| resource("Tet10 Jacobian work counter overflowed"))?;
        if *work > maximum_search_work {
            return Err(resource(format!(
                "Tet10 Jacobian certification exceeds its hard search-work limit of {maximum_search_work}"
            )));
        }
        if (*work == 4 || (*work).is_multiple_of(cancellation_check_interval))
            && cancellation.is_cancelled()
        {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::Cancelled,
                "cancelled",
            ));
        }
        let determinant = determinant_enclosure(coordinates, cell)?.scaled(orientation);
        if determinant.lower > 0.0 {
            continue;
        }
        if determinant.upper <= 0.0 {
            return Err(invalid(
                "Tet10 isoparametric mapping has a certified nonpositive Jacobian region",
            ));
        }
        if depth >= maximum_recursion_depth {
            return Err(invalid(format!(
                "Tet10 isoparametric Jacobian could not be certified positive by recursion depth {maximum_recursion_depth}"
            )));
        }
        let (left, right) = bisect(cell);
        pending.push((right, depth + 1));
        pending.push((left, depth + 1));
    }
    Ok(())
}

fn determinant_enclosure(
    coordinates: [[f64; 3]; 10],
    cell: [[f64; 4]; 4],
) -> Result<Interval, DelaunaySolverTopologyError> {
    let mut hull = [[Interval::empty(); 3]; 3];
    for barycentric in cell {
        let jacobian = jacobian_at(coordinates, barycentric);
        for row in 0..3 {
            for column in 0..3 {
                hull[row][column] = hull[row][column].hull(jacobian[row][column]);
            }
        }
    }
    let determinant = hull[0][0] * (hull[1][1] * hull[2][2] - hull[1][2] * hull[2][1])
        - hull[0][1] * (hull[1][0] * hull[2][2] - hull[1][2] * hull[2][0])
        + hull[0][2] * (hull[1][0] * hull[2][1] - hull[1][1] * hull[2][0]);
    if !determinant.lower.is_finite()
        || !determinant.upper.is_finite()
        || determinant.lower > determinant.upper
    {
        return Err(invalid(
            "Tet10 isoparametric Jacobian interval is not finite",
        ));
    }
    Ok(determinant)
}

fn jacobian_at(coordinates: [[f64; 3]; 10], barycentric: [f64; 4]) -> [[Interval; 3]; 3] {
    let barycentric = barycentric.map(Interval::exact);
    let mut result = [[Interval::zero(); 3]; 3];
    for node in 0..4 {
        let factor = barycentric[node] * Interval::exact(4.0) - Interval::exact(1.0);
        for reference_axis in 0..3 {
            let derivative =
                factor * Interval::exact(BARYCENTRIC_DERIVATIVES[node][reference_axis]);
            for (physical_axis, result_row) in result.iter_mut().enumerate() {
                result_row[reference_axis] = result_row[reference_axis]
                    + derivative * Interval::exact(coordinates[node][physical_axis]);
            }
        }
    }
    for (local_edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        for reference_axis in 0..3 {
            let derivative = Interval::exact(4.0)
                * (barycentric[right]
                    * Interval::exact(BARYCENTRIC_DERIVATIVES[left][reference_axis])
                    + barycentric[left]
                        * Interval::exact(BARYCENTRIC_DERIVATIVES[right][reference_axis]));
            for (physical_axis, result_row) in result.iter_mut().enumerate() {
                result_row[reference_axis] = result_row[reference_axis]
                    + derivative * Interval::exact(coordinates[4 + local_edge][physical_axis]);
            }
        }
    }
    result
}

fn bisect(mut cell: [[f64; 4]; 4]) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let mut selected = (0, 1);
    let mut maximum_length = -1.0;
    for left in 0..4 {
        for right in left + 1..4 {
            let length = (1..4)
                .map(|axis| {
                    let delta = cell[left][axis] - cell[right][axis];
                    delta * delta
                })
                .sum::<f64>();
            if length > maximum_length {
                maximum_length = length;
                selected = (left, right);
            }
        }
    }
    let midpoint =
        std::array::from_fn(|axis| cell[selected.0][axis] * 0.5 + cell[selected.1][axis] * 0.5);
    let mut right = cell;
    cell[selected.1] = midpoint;
    right[selected.0] = midpoint;
    (cell, right)
}

#[derive(Clone, Copy)]
struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    const fn exact(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    const fn zero() -> Self {
        Self::exact(0.0)
    }

    const fn empty() -> Self {
        Self {
            lower: f64::INFINITY,
            upper: f64::NEG_INFINITY,
        }
    }

    fn hull(self, other: Self) -> Self {
        Self {
            lower: self.lower.min(other.lower),
            upper: self.upper.max(other.upper),
        }
    }

    fn scaled(self, factor: f64) -> Self {
        if factor > 0.0 {
            self
        } else {
            Self {
                lower: -self.upper,
                upper: -self.lower,
            }
        }
    }
}

impl std::ops::Add for Interval {
    type Output = Self;

    fn add(self, right: Self) -> Self {
        Self {
            lower: next_down(self.lower + right.lower),
            upper: next_up(self.upper + right.upper),
        }
    }
}

impl std::ops::Sub for Interval {
    type Output = Self;

    fn sub(self, right: Self) -> Self {
        Self {
            lower: next_down(self.lower - right.upper),
            upper: next_up(self.upper - right.lower),
        }
    }
}

impl std::ops::Mul for Interval {
    type Output = Self;

    fn mul(self, right: Self) -> Self {
        let products = [
            self.lower * right.lower,
            self.lower * right.upper,
            self.upper * right.lower,
            self.upper * right.upper,
        ];
        Self {
            lower: next_down(products.into_iter().fold(f64::INFINITY, f64::min)),
            upper: next_up(products.into_iter().fold(f64::NEG_INFINITY, f64::max)),
        }
    }
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}

fn resource(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::ResourceLimit, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::NeverCancelled;

    fn straight() -> [[f64; 3]; 10] {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        [
            corners[0],
            corners[1],
            corners[2],
            corners[3],
            midpoint(corners[0], corners[1]),
            midpoint(corners[1], corners[2]),
            midpoint(corners[2], corners[0]),
            midpoint(corners[0], corners[3]),
            midpoint(corners[1], corners[3]),
            midpoint(corners[2], corners[3]),
        ]
    }

    #[test]
    fn interval_certificate_accepts_linear_mapping_and_rejects_curved_inversion() {
        let mut work = 0;
        certify_element(straight(), 10_000, 16, 1, &NeverCancelled, &mut work).unwrap();

        let mut inverted = straight();
        inverted[4] = [0.5, 0.0, 4.0];
        let mut work = 0;
        assert_eq!(
            certify_element(inverted, 100_000, 24, 1, &NeverCancelled, &mut work)
                .unwrap_err()
                .kind,
            DelaunaySolverTopologyErrorKind::InvalidMesh
        );
    }

    #[test]
    fn interval_certificate_enforces_work_and_cancellation() {
        let mut work = 0;
        assert_eq!(
            certify_element(straight(), 3, 16, 1, &NeverCancelled, &mut work)
                .unwrap_err()
                .kind,
            DelaunaySolverTopologyErrorKind::ResourceLimit
        );

        struct Cancelled;
        impl MeshingCancellationSignal for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let mut work = 0;
        assert_eq!(
            certify_element(straight(), 10_000, 16, 1, &Cancelled, &mut work)
                .unwrap_err()
                .kind,
            DelaunaySolverTopologyErrorKind::Cancelled
        );
    }

    fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| left[axis] * 0.5 + right[axis] * 0.5)
    }
}
