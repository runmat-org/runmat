use crate::cdt::{DelaunayConstraints, DelaunayVolumeNode};

const CANONICAL_COORDINATE_QUANTUM: f64 = 1.0 / 268_435_456.0;

/// Geometry-canonical ordering hints. Quantization affects insertion order only:
/// predicates, constraints, metric evaluation and published points retain the
/// original physical coordinates and construction authority.
pub(super) struct CanonicalVolumeFrame {
    origin_m: [f64; 3],
    axes: [[f64; 3]; 3],
    scale_m: f64,
}

impl CanonicalVolumeFrame {
    pub(super) fn from_constraints(constraints: &DelaunayConstraints) -> Option<Self> {
        if constraints.nodes.len() < 4 {
            return None;
        }
        let origin_m = compensated_average(
            constraints
                .nodes
                .iter()
                .map(|node| node.coordinates_m.canonical_value()),
        )?;
        let vectors = constraints
            .nodes
            .iter()
            .map(|node| subtract(node.coordinates_m.canonical_value(), origin_m))
            .collect::<Vec<_>>();
        let maximum_radius = vectors
            .iter()
            .map(|vector| dot(*vector, *vector).sqrt())
            .fold(0.0_f64, f64::max);
        if !maximum_radius.is_finite() || maximum_radius <= 0.0 {
            return None;
        }
        let scale_m = maximum_radius;
        let axes = canonical_principal_axes(&vectors, scale_m)?;
        Some(Self {
            origin_m,
            axes,
            scale_m,
        })
    }

    pub(super) fn compare_nodes(
        &self,
        left: &DelaunayVolumeNode,
        right: &DelaunayVolumeNode,
    ) -> std::cmp::Ordering {
        self.to_canonical(left.coordinates_m.canonical_value())
            .into_iter()
            .zip(self.to_canonical(right.coordinates_m.canonical_value()))
            .find_map(|(left, right)| (left != right).then(|| left.total_cmp(&right)))
            .unwrap_or_else(|| left.identity.cmp(&right.identity))
    }

    #[cfg(test)]
    pub(super) fn canonicalize_constraints(
        &self,
        constraints: &DelaunayConstraints,
    ) -> DelaunayConstraints {
        let mut canonical = constraints.clone();
        for node in &mut canonical.nodes {
            node.coordinates_m =
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    self.to_canonical(node.coordinates_m.canonical_value()),
                );
        }
        canonical
    }

    fn to_canonical(&self, point_m: [f64; 3]) -> [f64; 3] {
        let relative = subtract(point_m, self.origin_m);
        self.axes.map(|axis| {
            let value = dot(relative, axis) / self.scale_m;
            normalize_zero(
                (value / CANONICAL_COORDINATE_QUANTUM).round() * CANONICAL_COORDINATE_QUANTUM,
            )
        })
    }
}

fn canonical_principal_axes(vectors: &[[f64; 3]], scale: f64) -> Option<[[f64; 3]; 3]> {
    let covariance = covariance(vectors, scale)?;
    let eigenvectors = symmetric_eigenvectors(covariance)?;
    let mut best = None::<(Vec<[i64; 3]>, [[f64; 3]; 3])>;
    for permutation in permutations() {
        for signs in sign_combinations() {
            let axes = std::array::from_fn(|axis| {
                eigenvectors[permutation[axis]].map(|value| value * signs[axis])
            });
            if dot(cross(axes[0], axes[1]), axes[2]) <= 0.0 {
                continue;
            }
            let cloud = quantized_cloud(vectors, scale, axes);
            if best
                .as_ref()
                .is_none_or(|(best_cloud, _)| cloud < *best_cloud)
            {
                best = Some((cloud, axes));
            }
        }
    }
    best.map(|(_, axes)| axes)
}

fn covariance(vectors: &[[f64; 3]], scale: f64) -> Option<[[f64; 3]; 3]> {
    let mut sums = [[0.0; 3]; 3];
    let mut corrections = [[0.0; 3]; 3];
    for vector in vectors {
        let normalized = vector.map(|value| value / scale);
        for row in 0..3 {
            for column in row..3 {
                compensated_add(
                    &mut sums[row][column],
                    &mut corrections[row][column],
                    normalized[row] * normalized[column],
                );
            }
        }
    }
    let count = vectors.len() as f64;
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in row..3 {
            let value = (sums[row][column] + corrections[row][column]) / count;
            if !value.is_finite() {
                return None;
            }
            result[row][column] = value;
            result[column][row] = value;
        }
    }
    Some(result)
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    *correction += if sum.abs() >= value.abs() {
        (*sum - next) + value
    } else {
        (value - next) + *sum
    };
    *sum = next;
}

/// Deterministic Jacobi diagonalization of a real symmetric 3x3 matrix. The
/// returned vectors are rows so they can be used directly as frame axes.
fn symmetric_eigenvectors(mut matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let (p, q) = [(0, 1), (0, 2), (1, 2)].into_iter().max_by(|left, right| {
            matrix[left.0][left.1]
                .abs()
                .total_cmp(&matrix[right.0][right.1].abs())
        })?;
        let off_diagonal = matrix[p][q];
        if off_diagonal.abs() <= 64.0 * f64::EPSILON {
            break;
        }
        let angle = 0.5 * (2.0 * off_diagonal).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = angle.sin_cos();
        for row in &mut matrix {
            let left = row[p];
            let right = row[q];
            row[p] = cosine * left - sine * right;
            row[q] = sine * left + cosine * right;
        }
        for index in 0..3 {
            let upper = matrix[p][index];
            let lower = matrix[q][index];
            matrix[p][index] = cosine * upper - sine * lower;
            matrix[q][index] = sine * upper + cosine * lower;
        }
        for row in &mut basis {
            let left = row[p];
            let right = row[q];
            row[p] = cosine * left - sine * right;
            row[q] = sine * left + cosine * right;
        }
    }
    let axes: [Option<[f64; 3]>; 3] =
        std::array::from_fn(|axis| normalize([basis[0][axis], basis[1][axis], basis[2][axis]]));
    Some([axes[0]?, axes[1]?, axes[2]?])
}

fn quantized_cloud(vectors: &[[f64; 3]], scale: f64, axes: [[f64; 3]; 3]) -> Vec<[i64; 3]> {
    let mut cloud = vectors
        .iter()
        .map(|vector| {
            axes.map(|axis| {
                (dot(*vector, axis) / scale / CANONICAL_COORDINATE_QUANTUM).round() as i64
            })
        })
        .collect::<Vec<_>>();
    cloud.sort_unstable();
    cloud
}

fn permutations() -> [[usize; 3]; 6] {
    [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
}

fn sign_combinations() -> [[f64; 3]; 8] {
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ]
}

fn compensated_average(points: impl Iterator<Item = [f64; 3]>) -> Option<[f64; 3]> {
    let mut sum = [0.0; 3];
    let mut correction = [0.0; 3];
    let mut count = 0_u64;
    for point in points {
        count += 1;
        for axis in 0..3 {
            compensated_add(&mut sum[axis], &mut correction[axis], point[axis]);
        }
    }
    (count > 0).then(|| std::array::from_fn(|axis| (sum[axis] + correction[axis]) / count as f64))
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(vector, vector).sqrt();
    (length.is_finite() && length > 0.0).then(|| vector.map(|value| value / length))
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::StableDigest;

    #[test]
    fn geometry_frame_canonicalizes_rigid_units_order_and_identities() {
        let points = [
            [0.2, -0.4, 0.1],
            [1.3, 0.2, -0.5],
            [-0.7, 1.1, 0.8],
            [0.4, -0.2, 1.7],
        ];
        let constraints = |points: [[f64; 3]; 4], reversed: bool| DelaunayConstraints {
            nodes: {
                let mut nodes = points
                    .into_iter()
                    .enumerate()
                    .map(
                        |(index, coordinates_m)| crate::cdt::DelaunayConstraintNode {
                            identity: StableDigest::from_bytes(
                                [if reversed { 20 - index } else { index + 1 } as u8; 32],
                            ),
                            source_vertex_id: None,
                            coordinates_m: runmat_types::physical::Quantity::<
                                runmat_types::physical::kind::Position,
                                _,
                            >::new(coordinates_m),
                        },
                    )
                    .collect::<Vec<_>>();
                if reversed {
                    nodes.reverse();
                }
                nodes
            },
            segments: Vec::new(),
            facets: Vec::new(),
        };
        let transformed = points.map(|point| {
            [
                10.0 + 1.0e-3 * (0.36 * point[0] - 0.48 * point[1] + 0.8 * point[2]),
                -20.0 + 1.0e-3 * (0.8 * point[0] + 0.6 * point[1]),
                30.0 + 1.0e-3 * (-0.48 * point[0] + 0.64 * point[1] + 0.6 * point[2]),
            ]
        });
        let first = constraints(points, false);
        let second = constraints(transformed, true);
        let first_frame = CanonicalVolumeFrame::from_constraints(&first).unwrap();
        let second_frame = CanonicalVolumeFrame::from_constraints(&second).unwrap();
        let first = first_frame.canonicalize_constraints(&first);
        let second = second_frame.canonicalize_constraints(&second);
        let sorted = |constraints: DelaunayConstraints| {
            let mut points = constraints
                .nodes
                .iter()
                .map(|node| node.coordinates_m.canonical_value())
                .collect::<Vec<_>>();
            points.sort_by(|left, right| {
                left.iter()
                    .zip(right)
                    .find_map(|(left, right)| (left != right).then(|| left.total_cmp(right)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            points
        };
        assert_eq!(sorted(first), sorted(second));
    }

    #[test]
    fn canonical_ordering_retains_original_points_and_exact_planarity() {
        let points = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.3, 0.4, 1.0],
            [1.7, 0.8, 2.0],
        ];
        let constraints = DelaunayConstraints {
            nodes: points
                .into_iter()
                .enumerate()
                .map(
                    |(index, coordinates_m)| crate::cdt::DelaunayConstraintNode {
                        identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
                        source_vertex_id: None,
                        coordinates_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::Position,
                            _,
                        >::new(coordinates_m),
                    },
                )
                .collect(),
            segments: Vec::new(),
            facets: Vec::new(),
        };
        let frame = CanonicalVolumeFrame::from_constraints(&constraints).unwrap();
        let expected = constraints.volume_nodes();
        let options = crate::cdt::DelaunayPointSetOptions::default();
        let topology = crate::cdt::point_set::build_delaunay_volume_point_set_with_order(
            expected.clone(),
            options,
            &runmat_meshing_core::NeverCancelled,
            |left, right| frame.compare_nodes(left, right),
        )
        .unwrap();
        assert_eq!(topology.nodes, expected);
        assert_eq!(
            crate::cdt::topology::orient_materialized_volume_nodes([
                topology.nodes[0],
                topology.nodes[1],
                topology.nodes[2],
                topology.nodes[3],
            ])
            .unwrap(),
            runmat_meshing_core::quality::predicate::PredicateSign::Zero
        );
        crate::cdt::validate_delaunay_volume_topology(
            &topology,
            options.insertion,
            &runmat_meshing_core::NeverCancelled,
        )
        .unwrap();
    }
}
