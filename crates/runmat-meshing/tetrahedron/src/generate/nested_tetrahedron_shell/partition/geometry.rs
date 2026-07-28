use runmat_meshing_core::quality::predicate::Point3;

pub(super) fn sub(left: Point3, right: Point3) -> Point3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(super) fn cross(left: Point3, right: Point3) -> Point3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(super) fn dot(left: Point3, right: Point3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn norm(vector: Point3) -> f64 {
    dot(vector, vector).sqrt()
}

pub(super) fn scale(vector: Point3, factor: f64) -> Point3 {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}
