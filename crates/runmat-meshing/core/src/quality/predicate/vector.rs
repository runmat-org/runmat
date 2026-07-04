use super::types::Point3;

pub fn sub(left: Point3, right: Point3) -> Point3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub fn add(left: Point3, right: Point3) -> Point3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub fn scale(value: Point3, factor: f64) -> Point3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

pub fn dot(left: Point3, right: Point3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub fn cross(left: Point3, right: Point3) -> Point3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub fn norm_squared(value: Point3) -> f64 {
    dot(value, value)
}

pub fn norm(value: Point3) -> f64 {
    norm_squared(value).sqrt()
}

pub fn distance_squared(left: Point3, right: Point3) -> f64 {
    norm_squared(sub(left, right))
}

pub fn distance(left: Point3, right: Point3) -> f64 {
    distance_squared(left, right).sqrt()
}
