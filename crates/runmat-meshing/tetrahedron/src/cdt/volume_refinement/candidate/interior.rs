use super::{strictly_inside, DelaunayVolumeRefinementCandidateError};

const DYADIC_BARYCENTRIC_DENOMINATORS: [u16; 3] = [4, 8, 16];

/// Selection consumes only the needed prefix; validation retains the complete
/// independent reconstruction below. Each yielded index is the original index
/// after representability and exact-coordinate deduplication.
pub(super) struct CanonicalInteriorPoints {
    points: [[f64; 3]; 4],
    next_weight: usize,
    emitted: Vec<[f64; 3]>,
}
impl CanonicalInteriorPoints {
    pub(super) fn new(points: [[f64; 3]; 4]) -> Self {
        Self {
            points,
            next_weight: 0,
            emitted: Vec::new(),
        }
    }
}
impl Iterator for CanonicalInteriorPoints {
    type Item = Result<[f64; 3], DelaunayVolumeRefinementCandidateError>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(&(denominator, weights)) = ordered_weights().get(self.next_weight) {
            self.next_weight += 1;
            let base: [f64; 3] = std::array::from_fn(|axis| {
                self.points
                    .iter()
                    .zip(weights)
                    .map(|(point, weight)| point[axis] * f64::from(weight))
                    .sum::<f64>()
                    / f64::from(denominator)
            });
            let coordinates = base.map(|value| [value, value.next_down(), value.next_up()]);
            for first in coordinates[0] {
                for second in coordinates[1] {
                    for third in coordinates[2] {
                        let candidate = [first, second, third];
                        if self.emitted.contains(&candidate) {
                            continue;
                        }
                        match strictly_inside(self.points, candidate) {
                            Ok(true) => {
                                self.emitted.push(candidate);
                                return Some(Ok(candidate));
                            }
                            Ok(false) => {}
                            Err(error) => {
                                self.next_weight = usize::MAX;
                                return Some(Err(error));
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

fn ordered_weights() -> &'static [(u16, [u16; 4])] {
    static WEIGHTS: std::sync::OnceLock<Vec<(u16, [u16; 4])>> = std::sync::OnceLock::new();
    WEIGHTS.get_or_init(|| {
        let mut ordered = Vec::new();
        for denominator in DYADIC_BARYCENTRIC_DENOMINATORS {
            let mut weights = positive_compositions(denominator);
            weights.sort_by_key(|weights| {
                let minimum = *weights.iter().min().expect("four barycentric weights");
                let maximum = *weights.iter().max().expect("four barycentric weights");
                (maximum - minimum, *weights)
            });
            ordered.extend(weights.into_iter().map(|weights| (denominator, weights)));
        }
        ordered
    })
}

/// Returns a bounded deterministic sequence of representable points strictly inside a
/// tetrahedron. A dyadic barycentric lattice covers balanced, face-biased, edge-biased, and
/// vertex-biased points; ordering prefers balanced points before progressively stronger bias.
pub(in crate::cdt::volume_refinement) fn canonical_interior_points(
    points: [[f64; 3]; 4],
) -> Result<Vec<[f64; 3]>, DelaunayVolumeRefinementCandidateError> {
    let mut candidates = Vec::new();
    for denominator in DYADIC_BARYCENTRIC_DENOMINATORS {
        let mut weights = positive_compositions(denominator);
        weights.sort_by_key(|weights| {
            let minimum = *weights.iter().min().expect("four barycentric weights");
            let maximum = *weights.iter().max().expect("four barycentric weights");
            (maximum - minimum, *weights)
        });
        for weights in weights {
            let base: [f64; 3] = std::array::from_fn(|axis| {
                points
                    .iter()
                    .zip(weights)
                    .map(|(point, weight)| point[axis] * f64::from(weight))
                    .sum::<f64>()
                    / f64::from(denominator)
            });
            let coordinates = base.map(|value| [value, value.next_down(), value.next_up()]);
            'images: for first in coordinates[0] {
                for second in coordinates[1] {
                    for third in coordinates[2] {
                        let candidate = [first, second, third];
                        if !candidates.contains(&candidate) && strictly_inside(points, candidate)? {
                            candidates.push(candidate);
                            break 'images;
                        }
                    }
                }
            }
        }
    }
    Ok(candidates)
}

fn positive_compositions(total: u16) -> Vec<[u16; 4]> {
    let mut compositions = Vec::new();
    for first in 1..=(total - 3) {
        for second in 1..=(total - first - 2) {
            for third in 1..=(total - first - second - 1) {
                let fourth = total - first - second - third;
                compositions.push([first, second, third, fourth]);
            }
        }
    }
    compositions
}

#[cfg(test)]
mod tests {
    use super::{canonical_interior_points, positive_compositions};

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Lazy selection evaluates one initial lattice point while its full sequence matches
    //   complete reconstruction exactly across three scales, six coordinate orders and reflection.
    //   Point ordering and indices are checked; original-model quality and performance remain separate
    //   gates.
    #[test]
    fn lazy_interior_family_matches_complete_reconstruction_across_orientation_and_scale() {
        for exponent in [-80, 0, 80] {
            let scale = 2f64.powi(exponent);
            for reflection in [-1.0, 1.0] {
                for axes in [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ] {
                    let mut points = [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [2.0, 0.0, 1.0],
                        [3.0, 0.0, 0.0],
                    ]
                    .map(|point| {
                        let point = [
                            point[0] * scale,
                            point[1] * scale * reflection,
                            point[2] * scale,
                        ];
                        axes.map(|axis| point[axis])
                    });
                    points.sort_by(super::super::compare_points);
                    let expected = canonical_interior_points(points).unwrap();
                    assert!(
                        !expected.is_empty(),
                        "either source orientation has a strict centroid"
                    );
                    let mut lazy = super::CanonicalInteriorPoints::new(points);
                    assert_eq!(lazy.next().unwrap().unwrap(), expected[0]);
                    assert_eq!(lazy.emitted.len(), 1);
                    assert_eq!(
                        lazy.next_weight, 1,
                        "selection must not construct the unused lattice tail"
                    );
                    let mut actual = lazy.emitted.clone();
                    actual.extend(lazy.collect::<Result<Vec<_>, _>>().unwrap());
                    assert_eq!(
                        actual, expected,
                        "original point ordering and indices must remain exact"
                    );
                }
            }
        }
    }

    #[test]
    fn strict_containment_accepts_both_orientations_and_rejects_boundary_points() {
        for reflection in [-1.0, 1.0] {
            let points = [
                [0.0, 0.0, 0.0],
                [1.0, reflection, 0.0],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 0.0],
            ];
            let centroid = [1.5, reflection * 0.25, 0.25];
            assert!(super::strictly_inside(points, centroid).unwrap());
            assert!(!super::strictly_inside(points, points[0]).unwrap());
            assert!(!super::strictly_inside(points, [10.0; 3]).unwrap());
        }
        assert!(!super::strictly_inside([[0.0; 3]; 4], [0.0; 3]).unwrap());
    }

    #[test]
    fn dyadic_lattice_is_bounded_balanced_first_and_covers_edge_bias() {
        assert_eq!(positive_compositions(4).len(), 1);
        assert_eq!(positive_compositions(8).len(), 35);
        assert_eq!(positive_compositions(16).len(), 455);

        let points = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let candidates = canonical_interior_points(points).unwrap();
        assert_eq!(candidates.first().copied(), Some([0.25, 0.25, 0.25]));
        assert!(candidates.len() <= 491);
        assert!(candidates.contains(&[0.125, 0.375, 0.125]));
    }
}
