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
