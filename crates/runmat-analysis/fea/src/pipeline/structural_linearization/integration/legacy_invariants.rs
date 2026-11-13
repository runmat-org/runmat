//! Preserve the geometric-tangent invariants formerly tested on separate element helpers.

use super::*;

#[test]
fn both_orders_preserve_sign_scaling_symmetry_and_translation_nullspace() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (element, coordinates) = super::tests::element(order);
        // Positive definite constant strain with nonzero shear exercises every tensor entry.
        let displacement: Vec<_> = coordinates
            .iter()
            .map(|&[x, y, z]| {
                [
                    0.03 * x + 0.002 * y,
                    0.02 * y + 0.003 * z,
                    0.01 * z + 0.001 * x,
                ]
            })
            .collect();
        let evaluate = |factor: f64| {
            let scaled: Vec<_> = displacement.iter().map(|v| v.map(|x| factor * x)).collect();
            integrate::<30>(&element, &scaled, order, &coordinates, false).unwrap()
        };
        let tension = evaluate(1.0);
        let compression = evaluate(-1.0);
        let doubled = evaluate(2.0);
        let count = coordinates.len() * 3;
        assert!(tension.matrix[0][0] > 0.0);
        for row in 0..count {
            for column in 0..count {
                let value = tension.matrix[row][column];
                assert!((value - tension.matrix[column][row]).abs() < 1e-12);
                assert!((compression.matrix[row][column] + value).abs() < 1e-12);
                assert!((doubled.matrix[row][column] - 2.0 * value).abs() < 1e-12);
                if row % 3 != column % 3 {
                    assert_eq!(value, 0.0);
                }
            }
            for axis in 0..3 {
                let residual: f64 = (0..coordinates.len())
                    .map(|node| tension.matrix[row][3 * node + axis])
                    .sum();
                assert!(residual.abs() < 1e-12, "{order:?}: {residual}");
            }
        }
    }
}
