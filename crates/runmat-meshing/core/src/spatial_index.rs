use serde::{Deserialize, Serialize};

use crate::predicate::{distance_squared, Point3, Triangle3};
use crate::tolerance::MeshingTolerance;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSpatialIndex<T> {
    entries: Vec<SpatialEntry<T>>,
}

impl<T> Default for LinearSpatialIndex<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<T> LinearSpatialIndex<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_entries(entries: Vec<SpatialEntry<T>>) -> Self {
        Self { entries }
    }

    pub fn insert(&mut self, bounds: Aabb3, payload: T) {
        self.entries.push(SpatialEntry { bounds, payload });
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[SpatialEntry<T>] {
        &self.entries
    }

    pub fn query_point(&self, point: Point3) -> impl Iterator<Item = &SpatialEntry<T>> {
        self.entries
            .iter()
            .filter(move |entry| entry.bounds.contains_point(point))
    }

    pub fn query_bounds(&self, bounds: Aabb3) -> impl Iterator<Item = &SpatialEntry<T>> {
        self.entries
            .iter()
            .filter(move |entry| entry.bounds.intersects(bounds))
    }

    pub fn nearest_by_center(&self, point: Point3) -> Option<&SpatialEntry<T>> {
        self.entries.iter().min_by(|left, right| {
            distance_squared(left.bounds.center_m(), point)
                .total_cmp(&distance_squared(right.bounds.center_m(), point))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_contains_and_intersects_points() {
        let bounds = Aabb3::from_points(&[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]).unwrap();

        assert!(bounds.contains_point([0.5, 1.0, 2.0]));
        assert!(!bounds.contains_point([2.0, 1.0, 2.0]));
        assert!(bounds.intersects(Aabb3::from_points(&[[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]]).unwrap()));
        assert!(
            !bounds.intersects(Aabb3::from_points(&[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]).unwrap())
        );
    }

    #[test]
    fn linear_spatial_index_queries_deterministically() {
        let mut index = LinearSpatialIndex::new();
        index.insert(
            Aabb3::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).unwrap(),
            "left",
        );
        index.insert(
            Aabb3::from_points(&[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]]).unwrap(),
            "right",
        );

        let hits = index
            .query_point([0.5, 0.5, 0.5])
            .map(|entry| entry.payload)
            .collect::<Vec<_>>();
        assert_eq!(hits, vec!["left"]);
        assert_eq!(
            index
                .nearest_by_center([2.8, 0.5, 0.5])
                .map(|entry| entry.payload),
            Some("right")
        );
    }
}
