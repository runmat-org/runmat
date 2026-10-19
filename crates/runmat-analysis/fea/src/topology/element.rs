pub(super) fn tetrahedron_geometry(coordinates: [[f64; 3]; 4]) -> Option<(f64, [f64; 3], f64)> {
    let ab = subtract(coordinates[1], coordinates[0]);
    let ac = subtract(coordinates[2], coordinates[0]);
    let ad = subtract(coordinates[3], coordinates[0]);
    let signed_volume = dot(ab, cross(ac, ad)) / 6.0;
    if !signed_volume.is_finite() || signed_volume <= 0.0 {
        return None;
    }
    let centroid =
        std::array::from_fn(|axis| coordinates.iter().map(|point| point[axis]).sum::<f64>() / 4.0);
    let characteristic_length = (6.0 * signed_volume).cbrt();
    Some((signed_volume, centroid, characteristic_length))
}

pub(super) fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(super) fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(super) fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

pub(super) fn invert(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c00 = matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1];
    let c01 = -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]);
    let c02 = matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0];
    let c10 = -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]);
    let c11 = matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0];
    let c12 = -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]);
    let c20 = matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1];
    let c21 = -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]);
    let c22 = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let determinant = matrix[0][0] * c00 + matrix[0][1] * c01 + matrix[0][2] * c02;
    if !determinant.is_finite() || determinant <= 0.0 {
        return None;
    }
    let inverse_determinant = determinant.recip();
    Some([
        [
            c00 * inverse_determinant,
            c10 * inverse_determinant,
            c20 * inverse_determinant,
        ],
        [
            c01 * inverse_determinant,
            c11 * inverse_determinant,
            c21 * inverse_determinant,
        ],
        [
            c02 * inverse_determinant,
            c12 * inverse_determinant,
            c22 * inverse_determinant,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::invert;

    #[test]
    fn inverse_reproduces_identity_on_both_sides_for_nonsymmetric_geometry() {
        // Unequal m20/m21 and nonzero m02 expose the off-diagonal cofactor.
        let matrices = [
            [[2.0, 1.0, 3.0], [0.0, 4.0, 1.0], [1.0, 2.0, 5.0]],
            [[0.0, -1.0, 1.0], [1.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ];
        for matrix in matrices {
            let inverse = invert(matrix).unwrap();
            for (left, right) in [(matrix, inverse), (inverse, matrix)] {
                for (row, values) in left.iter().enumerate() {
                    for column in 0..3 {
                        let actual = (0..3).map(|k| values[k] * right[k][column]).sum::<f64>();
                        let expected = if row == column { 1.0 } else { 0.0 };
                        assert!((actual - expected).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
