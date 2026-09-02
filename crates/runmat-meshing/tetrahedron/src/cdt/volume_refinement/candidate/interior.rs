use super::{strictly_inside, DelaunayVolumeRefinementCandidateError};

const DYADIC_BARYCENTRIC_DENOMINATORS: [u16; 3] = [4, 8, 16];

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
