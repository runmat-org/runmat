pub(super) fn add_scaled<const N: usize>(
    origin: &[f64; N],
    first: &[f64; N],
    first_scale: f64,
    second: &[f64; N],
    second_scale: f64,
) -> [f64; N] {
    std::array::from_fn(|index| {
        origin[index] + first[index] * first_scale + second[index] * second_scale
    })
}

pub(super) fn scale<const N: usize>(value: &[f64; N], factor: f64) -> [f64; N] {
    std::array::from_fn(|index| value[index] * factor)
}

pub(super) fn subtract<const N: usize>(left: &[f64; N], right: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|index| left[index] - right[index])
}

pub(super) fn dot<const N: usize>(left: &[f64; N], right: &[f64; N]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub(super) fn norm<const N: usize>(value: &[f64; N]) -> f64 {
    dot(value, value).sqrt()
}

pub(super) fn distance<const N: usize>(left: &[f64; N], right: &[f64; N]) -> f64 {
    norm(&subtract(left, right))
}

pub(super) fn normalize<const N: usize>(value: &[f64; N]) -> Option<[f64; N]> {
    let magnitude = norm(value);
    (magnitude.is_finite() && magnitude > 0.0).then(|| scale(value, magnitude.recip()))
}

pub(super) fn cross(left: &[f64; 3], right: &[f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
