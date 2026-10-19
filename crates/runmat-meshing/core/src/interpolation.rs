//! Canonical reference-element interpolation shared by geometry and numerical consumers.
use crate::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

/// Tet10 Lagrange weights: corners0..4 then edges01,12,20,03,13,23.
/// This polynomial evaluator does not locate a point or validate a physical geometry.
/// Callers own reference-coordinate admission and the supported geometry envelope.
pub fn tetrahedron10_lagrange_weights(barycentric: [f64; 4]) -> [f64; 10] {
    let mut values = [0.0; 10];
    for node in 0..4 {
        values[node] = barycentric[node] * (2.0 * barycentric[node] - 1.0);
    }
    for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        values[4 + edge] = 4.0 * barycentric[left] * barycentric[right];
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lagrange_basis_preserves_nodes_partition_and_quadratic_polynomials() {
        let nodes: [[f64; 4]; 10] = std::array::from_fn(|node| {
            let mut b = [0.0; 4];
            if node < 4 {
                b[node] = 1.0;
            } else {
                let [a, c] = TETRAHEDRON_MIDSIDE_EDGE_CORNERS[node - 4];
                b[a] = 0.5;
                b[c] = 0.5;
            }
            b
        });
        for (i, b) in nodes.into_iter().enumerate() {
            for (j, w) in tetrahedron10_lagrange_weights(b).into_iter().enumerate() {
                assert_eq!(w, if i == j { 1.0 } else { 0.0 });
            }
        }
        for b in [
            [0.25; 4],
            [0.625, 0.125, 0.125, 0.125],
            [0.0, 0.2, 0.3, 0.5],
        ] {
            let weights = tetrahedron10_lagrange_weights(b);
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
            for a in 0..4 {
                for c in 0..4 {
                    let actual = weights
                        .iter()
                        .zip(nodes)
                        .map(|(w, n)| w * n[a] * n[c])
                        .sum::<f64>();
                    assert!((actual - b[a] * b[c]).abs() < 1.0e-14);
                }
            }
        }
        let centroid = tetrahedron10_lagrange_weights([0.25; 4]);
        assert_eq!(&centroid[..4], &[-0.125; 4]);
        assert_eq!(&centroid[4..], &[0.25; 6]);
    }
}
