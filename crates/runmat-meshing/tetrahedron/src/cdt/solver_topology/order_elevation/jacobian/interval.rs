use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

use crate::cdt::solver_topology::DelaunaySolverTopologyError;

use super::{invalid, BARYCENTRIC_DERIVATIVES};

pub(super) fn has_certified_nonpositive_vertex(
    coordinates: [[f64; 3]; 10],
    cell: [[f64; 4]; 4],
    orientation: f64,
) -> bool {
    cell.into_iter().any(|barycentric| {
        let jacobian = jacobian_at(coordinates, barycentric);
        determinant_columns([
            column(&jacobian, 0),
            column(&jacobian, 1),
            column(&jacobian, 2),
        ])
        .scaled(orientation)
        .upper
            <= 0.0
    })
}

pub(super) fn determinant_enclosure(
    coordinates: [[f64; 3]; 10],
    cell: [[f64; 4]; 4],
) -> Result<Interval, DelaunaySolverTopologyError> {
    let entrywise = entrywise_determinant_enclosure(coordinates, cell);
    if entrywise.lower > 0.0 || entrywise.upper <= 0.0 {
        return validate_finite_enclosure(entrywise);
    }

    // A Tet10 Jacobian is affine in the reference coordinates, so its determinant is a cubic
    // polynomial. Its degree-three Bernstein coefficients over this cell are the mixed
    // determinants of the four vertex Jacobians. Retaining that dependency is substantially
    // sharper than independently hulling the nine Jacobian entries.
    let jacobians = cell.map(|barycentric| jacobian_at(coordinates, barycentric));
    let mut determinant = Interval::empty();
    for first in 0..4 {
        for second in first..4 {
            for third in second..4 {
                determinant = determinant.hull(mixed_determinant([
                    &jacobians[first],
                    &jacobians[second],
                    &jacobians[third],
                ]));
            }
        }
    }
    validate_finite_enclosure(determinant)
}

fn entrywise_determinant_enclosure(coordinates: [[f64; 3]; 10], cell: [[f64; 4]; 4]) -> Interval {
    let mut hull = [[Interval::empty(); 3]; 3];
    for barycentric in cell {
        let jacobian = jacobian_at(coordinates, barycentric);
        for row in 0..3 {
            for column in 0..3 {
                hull[row][column] = hull[row][column].hull(jacobian[row][column]);
            }
        }
    }
    determinant_columns([column(&hull, 0), column(&hull, 1), column(&hull, 2)])
}

fn validate_finite_enclosure(
    determinant: Interval,
) -> Result<Interval, DelaunaySolverTopologyError> {
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

pub(super) fn mixed_determinant(matrices: [&[[Interval; 3]; 3]; 3]) -> Interval {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    PERMUTATIONS
        .into_iter()
        .map(|assignment| {
            determinant_columns([
                column(matrices[assignment[0]], 0),
                column(matrices[assignment[1]], 1),
                column(matrices[assignment[2]], 2),
            ])
        })
        .fold(Interval::zero(), |sum, term| sum + term)
        .divided_by_positive_integer(6.0)
}

pub(super) fn column(matrix: &[[Interval; 3]; 3], column: usize) -> [Interval; 3] {
    std::array::from_fn(|row| matrix[row][column])
}

pub(super) fn determinant_columns(columns: [[Interval; 3]; 3]) -> Interval {
    columns[0][0] * (columns[1][1] * columns[2][2] - columns[2][1] * columns[1][2])
        - columns[1][0] * (columns[0][1] * columns[2][2] - columns[2][1] * columns[0][2])
        + columns[2][0] * (columns[0][1] * columns[1][2] - columns[1][1] * columns[0][2])
}

pub(super) fn jacobian_at(
    coordinates: [[f64; 3]; 10],
    barycentric: [f64; 4],
) -> [[Interval; 3]; 3] {
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

#[derive(Clone, Copy)]
pub(super) struct Interval {
    pub(super) lower: f64,
    pub(super) upper: f64,
}

impl Interval {
    pub(super) const fn exact(value: f64) -> Self {
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

    pub(super) fn scaled(self, factor: f64) -> Self {
        if factor > 0.0 {
            self
        } else {
            Self {
                lower: -self.upper,
                upper: -self.lower,
            }
        }
    }

    fn divided_by_positive_integer(self, divisor: f64) -> Self {
        debug_assert!(divisor.is_finite() && divisor > 0.0 && divisor.fract() == 0.0);
        Self {
            lower: next_down(self.lower / divisor),
            upper: next_up(self.upper / divisor),
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
