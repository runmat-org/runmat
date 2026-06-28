use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniformGridSpatialIndex<T> {
    entries: Vec<SpatialEntry<T>>,
    bounds: Aabb3,
    dimensions: [usize; 3],
    cells: BTreeMap<[usize; 3], Vec<usize>>,
}

impl<T> UniformGridSpatialIndex<T> {
    pub fn from_entries(entries: Vec<SpatialEntry<T>>) -> Self {
        if entries.is_empty() {
            return Self {
                entries,
                bounds: Aabb3 {
                    min_m: [0.0, 0.0, 0.0],
                    max_m: [0.0, 0.0, 0.0],
                },
                dimensions: [1, 1, 1],
                cells: BTreeMap::new(),
            };
        }
        let bounds = entries
            .iter()
            .map(|entry| entry.bounds)
            .reduce(|left, right| Aabb3 {
                min_m: [
                    left.min_m[0].min(right.min_m[0]),
                    left.min_m[1].min(right.min_m[1]),
                    left.min_m[2].min(right.min_m[2]),
                ],
                max_m: [
                    left.max_m[0].max(right.max_m[0]),
                    left.max_m[1].max(right.max_m[1]),
                    left.max_m[2].max(right.max_m[2]),
                ],
            })
            .unwrap();
        let dimensions = grid_dimensions(bounds, entries.len());
        let mut cells = BTreeMap::<[usize; 3], Vec<usize>>::new();
        for (entry_index, entry) in entries.iter().enumerate() {
            let min_cell = cell_for_point(bounds, dimensions, entry.bounds.min_m);
            let max_cell = cell_for_point(bounds, dimensions, entry.bounds.max_m);
            for x in min_cell[0]..=max_cell[0] {
                for y in min_cell[1]..=max_cell[1] {
                    for z in min_cell[2]..=max_cell[2] {
                        cells.entry([x, y, z]).or_default().push(entry_index);
                    }
                }
            }
        }
        Self {
            entries,
            bounds,
            dimensions,
            cells,
        }
    }

    pub fn entries(&self) -> &[SpatialEntry<T>] {
        &self.entries
    }

    pub fn bounds(&self) -> Aabb3 {
        self.bounds
    }

    pub fn query_point(&self, point: Point3) -> Vec<&SpatialEntry<T>> {
        if self.entries.is_empty() || !self.bounds.contains_point(point) {
            return Vec::new();
        }
        let cell = cell_for_point(self.bounds, self.dimensions, point);
        self.cells
            .get(&cell)
            .into_iter()
            .flatten()
            .filter_map(|index| self.entries.get(*index))
            .filter(|entry| entry.bounds.contains_point(point))
            .collect()
    }

    pub fn query_bounds(&self, bounds: Aabb3) -> Vec<&SpatialEntry<T>> {
        if self.entries.is_empty() || !self.bounds.intersects(bounds) {
            return Vec::new();
        }
        let min_cell = cell_for_point(self.bounds, self.dimensions, bounds.min_m);
        let max_cell = cell_for_point(self.bounds, self.dimensions, bounds.max_m);
        let mut entry_indices = BTreeSet::<usize>::new();
        for x in min_cell[0]..=max_cell[0] {
            for y in min_cell[1]..=max_cell[1] {
                for z in min_cell[2]..=max_cell[2] {
                    if let Some(indices) = self.cells.get(&[x, y, z]) {
                        entry_indices.extend(indices.iter().copied());
                    }
                }
            }
        }
        entry_indices
            .into_iter()
            .filter_map(|index| self.entries.get(index))
            .filter(|entry| entry.bounds.intersects(bounds))
            .collect()
    }

    pub fn nearest_by_center(&self, point: Point3) -> Option<&SpatialEntry<T>> {
        self.entries.iter().min_by(|left, right| {
            distance_squared(left.bounds.center_m(), point)
                .total_cmp(&distance_squared(right.bounds.center_m(), point))
        })
    }
}

fn grid_dimensions(bounds: Aabb3, entry_count: usize) -> [usize; 3] {
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

fn cell_for_point(bounds: Aabb3, dimensions: [usize; 3], point: Point3) -> [usize; 3] {
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

    #[test]
    fn uniform_grid_spatial_index_filters_point_and_bounds_queries() {
        let entries = vec![
            SpatialEntry {
                bounds: Aabb3::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).unwrap(),
                payload: "left",
            },
            SpatialEntry {
                bounds: Aabb3::from_points(&[[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]]).unwrap(),
                payload: "right",
            },
        ];
        let index = UniformGridSpatialIndex::from_entries(entries);

        let point_hits = index
            .query_point([0.5, 0.5, 0.5])
            .into_iter()
            .map(|entry| entry.payload)
            .collect::<Vec<_>>();
        assert_eq!(point_hits, vec!["left"]);
        let bounds_hits = index
            .query_bounds(Aabb3::from_points(&[[2.5, 0.5, 0.5], [4.0, 0.6, 0.6]]).unwrap())
            .into_iter()
            .map(|entry| entry.payload)
            .collect::<Vec<_>>();
        assert_eq!(bounds_hits, vec!["right"]);
    }
}
