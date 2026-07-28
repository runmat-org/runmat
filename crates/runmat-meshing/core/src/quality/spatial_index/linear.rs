use serde::{Deserialize, Serialize};

use crate::quality::predicate::{distance_squared, Point3};

use super::types::{Aabb3, SpatialEntry};

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
