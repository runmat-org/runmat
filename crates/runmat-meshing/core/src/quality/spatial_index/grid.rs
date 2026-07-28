use crate::quality::predicate::Point3;

use super::types::Aabb3;

pub(super) fn grid_dimensions(bounds: Aabb3, entry_count: usize) -> [usize; 3] {
    let target_axis_count = ((entry_count as f64).cbrt().ceil() as usize).clamp(1, 64);
    let spans = [
        bounds.max_m[0] - bounds.min_m[0],
        bounds.max_m[1] - bounds.min_m[1],
        bounds.max_m[2] - bounds.min_m[2],
    ];
    let max_span = spans.into_iter().fold(0.0_f64, f64::max);
    if max_span <= f64::EPSILON {
        return [1, 1, 1];
    }
    spans.map(|span| {
        ((span / max_span) * target_axis_count as f64)
            .ceil()
            .max(1.0) as usize
    })
}

pub(super) fn cell_for_point(bounds: Aabb3, dimensions: [usize; 3], point: Point3) -> [usize; 3] {
    let mut cell = [0_usize; 3];
    for axis in 0..3 {
        let span = bounds.max_m[axis] - bounds.min_m[axis];
        if span <= f64::EPSILON || dimensions[axis] <= 1 {
            cell[axis] = 0;
            continue;
        }
        let t = ((point[axis] - bounds.min_m[axis]) / span).clamp(0.0, 1.0);
        cell[axis] = ((t * dimensions[axis] as f64).floor() as usize).min(dimensions[axis] - 1);
    }
    cell
}

pub(super) fn ray_bounds_intersection(
    origin: Point3,
    direction: Point3,
    bounds: Aabb3,
) -> Option<(f64, f64)> {
    let mut ray_min = f64::NEG_INFINITY;
    let mut ray_max = f64::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() <= f64::EPSILON {
            if origin[axis] < bounds.min_m[axis] || origin[axis] > bounds.max_m[axis] {
                return None;
            }
            continue;
        }
        let t0 = (bounds.min_m[axis] - origin[axis]) / direction[axis];
        let t1 = (bounds.max_m[axis] - origin[axis]) / direction[axis];
        ray_min = ray_min.max(t0.min(t1));
        ray_max = ray_max.min(t0.max(t1));
        if ray_min > ray_max {
            return None;
        }
    }
    Some((ray_min, ray_max))
}

pub(super) fn add_point(left: Point3, right: Point3) -> Point3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub(super) fn scale_point(point: Point3, scale: f64) -> Point3 {
    [point[0] * scale, point[1] * scale, point[2] * scale]
}
