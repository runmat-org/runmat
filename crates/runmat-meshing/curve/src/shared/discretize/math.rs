use runmat_meshing_core::MetricTensor3;

pub(super) fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub(super) fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(super) fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

pub(super) fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn norm(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

pub(super) fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(value);
    (length.is_finite() && length > 0.0).then(|| scale(value, 1.0 / length))
}

pub(super) fn tangent_angle(left: [f64; 3], right: [f64; 3]) -> f64 {
    dot(left, right).clamp(-1.0, 1.0).acos()
}

pub(super) fn point_segment_distance(point: [f64; 3], left: [f64; 3], right: [f64; 3]) -> f64 {
    let chord = sub(right, left);
    let squared = dot(chord, chord);
    if squared == 0.0 {
        return norm(sub(point, left));
    }
    let fraction = (dot(sub(point, left), chord) / squared).clamp(0.0, 1.0);
    norm(sub(point, add(left, scale(chord, fraction))))
}

pub(super) fn metric_length(delta: [f64; 3], metric: MetricTensor3) -> f64 {
    let [x, y, z] = delta;
    (metric.xx * x * x
        + metric.yy * y * y
        + metric.zz * z * z
        + 2.0 * metric.xy * x * y
        + 2.0 * metric.xz * x * z
        + 2.0 * metric.yz * y * z)
        .max(0.0)
        .sqrt()
}

pub(super) fn average_metric(left: MetricTensor3, right: MetricTensor3) -> MetricTensor3 {
    MetricTensor3 {
        xx: (left.xx + right.xx) * 0.5,
        yy: (left.yy + right.yy) * 0.5,
        zz: (left.zz + right.zz) * 0.5,
        xy: (left.xy + right.xy) * 0.5,
        xz: (left.xz + right.xz) * 0.5,
        yz: (left.yz + right.yz) * 0.5,
    }
}
