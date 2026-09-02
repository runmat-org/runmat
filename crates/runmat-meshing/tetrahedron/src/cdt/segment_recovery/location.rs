pub(in crate::cdt) fn coordinates_share_representable_location(
    left: [f64; 3],
    right: [f64; 3],
) -> bool {
    let scale = left
        .into_iter()
        .chain(right)
        .map(f64::abs)
        .fold(1.0_f64, f64::max);
    let tolerance = representable_location_tolerance(scale);
    left.into_iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

fn representable_location_tolerance(scale_m: f64) -> f64 {
    128.0 * f64::EPSILON * scale_m.max(1.0)
}
