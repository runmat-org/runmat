pub(super) fn power_f64(real: f64, imaginary: f64) -> (f64, f64) {
    if imaginary == 0.0 {
        return (real.exp2(), 0.0);
    }
    let scale = (real * std::f64::consts::LN_2).exp();
    let angle = imaginary * std::f64::consts::LN_2;
    (scale * angle.cos(), scale * angle.sin())
}

pub(super) fn power_f32(real: f32, imaginary: f32) -> (f32, f32) {
    if imaginary == 0.0 {
        return (real.exp2(), 0.0);
    }
    let scale = (real * std::f32::consts::LN_2).exp();
    let angle = imaginary * std::f32::consts::LN_2;
    (scale * angle.cos(), scale * angle.sin())
}

pub(super) const fn multiply_f64(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

pub(super) const fn multiply_f32(left: (f32, f32), right: (f32, f32)) -> (f32, f32) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}
