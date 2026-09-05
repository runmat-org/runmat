pub(super) fn f64(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.hypot(right)
    }
}

pub(super) fn f32(left: f32, right: f32) -> f32 {
    if left.is_nan() || right.is_nan() {
        f32::NAN
    } else {
        left.hypot(right)
    }
}

pub(in super::super) fn complex_magnitude(real: f64, imaginary: f64) -> f64 {
    f64(real, imaginary)
}
