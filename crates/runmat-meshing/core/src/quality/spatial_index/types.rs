use serde::{Deserialize, Serialize};

use crate::quality::{
    predicate::{Point3, Triangle3},
    tolerance::MeshingTolerance,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb3 {
    pub min_m: Point3,
    pub max_m: Point3,
}

impl Aabb3 {
    pub fn from_points(points: &[Point3]) -> Option<Self> {
        let first = *points.first()?;
        if first.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut min_m = first;
        let mut max_m = first;
        for point in points.iter().skip(1) {
            if point.iter().any(|value| !value.is_finite()) {
                return None;
            }
            for axis in 0..3 {
                min_m[axis] = min_m[axis].min(point[axis]);
                max_m[axis] = max_m[axis].max(point[axis]);
            }
        }
        Some(Self { min_m, max_m })
    }

    pub fn from_triangle(triangle: Triangle3) -> Self {
        Self::from_points(&triangle).expect("triangle points are nonempty")
    }

    pub fn expanded(self, tolerance: MeshingTolerance) -> Self {
        let span = self.max_span_m();
        let epsilon = tolerance.length_epsilon(span);
        Self {
            min_m: [
                self.min_m[0] - epsilon,
                self.min_m[1] - epsilon,
                self.min_m[2] - epsilon,
            ],
            max_m: [
                self.max_m[0] + epsilon,
                self.max_m[1] + epsilon,
                self.max_m[2] + epsilon,
            ],
        }
    }

    pub fn contains_point(self, point: Point3) -> bool {
        (0..3).all(|axis| point[axis] >= self.min_m[axis] && point[axis] <= self.max_m[axis])
    }

    pub fn intersects(self, other: Self) -> bool {
        (0..3).all(|axis| {
            self.min_m[axis] <= other.max_m[axis] && self.max_m[axis] >= other.min_m[axis]
        })
    }

    pub fn center_m(self) -> Point3 {
        [
            (self.min_m[0] + self.max_m[0]) * 0.5,
            (self.min_m[1] + self.max_m[1]) * 0.5,
            (self.min_m[2] + self.max_m[2]) * 0.5,
        ]
    }

    pub fn max_span_m(self) -> f64 {
        (0..3)
            .map(|axis| self.max_m[axis] - self.min_m[axis])
            .fold(0.0_f64, f64::max)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialEntry<T> {
    pub bounds: Aabb3,
    pub payload: T,
}
