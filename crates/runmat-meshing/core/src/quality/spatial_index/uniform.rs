mod cooperative;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::quality::predicate::{distance_squared, Point3};

use super::{
    grid::{add_point, cell_for_point, grid_dimensions, ray_bounds_intersection, scale_point},
    types::{Aabb3, SpatialEntry},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniformGridSpatialIndex<T> {
    entries: Vec<SpatialEntry<T>>,
    bounds: Aabb3,
    dimensions: [usize; 3],
    cells: BTreeMap<[usize; 3], Vec<usize>>,
}

impl<T> UniformGridSpatialIndex<T> {
    pub fn from_entries(entries: Vec<SpatialEntry<T>>) -> Self {
        Self::from_entries_with_checkpoint(entries, &mut || Ok::<_, std::convert::Infallible>(()))
            .unwrap_or_else(|error| match error {})
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
        self.query_bounds_with_checkpoint(bounds, &mut || Ok::<_, std::convert::Infallible>(()))
            .unwrap_or_else(|error| match error {})
    }

    pub fn query_ray(&self, origin: Point3, direction: Point3) -> Vec<&SpatialEntry<T>> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let Some((ray_min, ray_max)) = ray_bounds_intersection(origin, direction, self.bounds)
        else {
            return Vec::new();
        };
        if ray_max < 0.0 {
            return Vec::new();
        }

        let mut cell = cell_for_point(
            self.bounds,
            self.dimensions,
            add_point(origin, scale_point(direction, ray_min.max(0.0))),
        );
        let mut step = [0_i8; 3];
        let mut next_t = [f64::INFINITY; 3];
        let mut delta_t = [f64::INFINITY; 3];
        for axis in 0..3 {
            let span = self.bounds.max_m[axis] - self.bounds.min_m[axis];
            if span <= f64::EPSILON
                || self.dimensions[axis] <= 1
                || direction[axis].abs() <= f64::EPSILON
            {
                continue;
            }
            let cell_size = span / self.dimensions[axis] as f64;
            if direction[axis] > 0.0 {
                step[axis] = 1;
                let boundary = self.bounds.min_m[axis] + (cell[axis] + 1) as f64 * cell_size;
                next_t[axis] = (boundary - origin[axis]) / direction[axis];
                delta_t[axis] = cell_size / direction[axis];
            } else {
                step[axis] = -1;
                let boundary = self.bounds.min_m[axis] + cell[axis] as f64 * cell_size;
                next_t[axis] = (boundary - origin[axis]) / direction[axis];
                delta_t[axis] = cell_size / direction[axis].abs();
            }
        }

        let mut current_t = ray_min.max(0.0);
        let mut entry_indices = BTreeSet::<usize>::new();
        while current_t <= ray_max + f64::EPSILON {
            if let Some(indices) = self.cells.get(&cell) {
                entry_indices.extend(indices.iter().copied().filter(|index| {
                    self.entries.get(*index).is_some_and(|entry| {
                        ray_bounds_intersection(origin, direction, entry.bounds).is_some()
                    })
                }));
            }

            let crossing_t = next_t.into_iter().fold(f64::INFINITY, f64::min);
            if !crossing_t.is_finite() || crossing_t > ray_max {
                break;
            }
            current_t = crossing_t;
            let mut advanced = false;
            for axis in 0..3 {
                if (next_t[axis] - crossing_t).abs() > f64::EPSILON {
                    continue;
                }
                if step[axis] > 0 {
                    cell[axis] += 1;
                    if cell[axis] >= self.dimensions[axis] {
                        return self.entries_for_indices(entry_indices);
                    }
                } else if step[axis] < 0 {
                    if cell[axis] == 0 {
                        return self.entries_for_indices(entry_indices);
                    }
                    cell[axis] -= 1;
                }
                next_t[axis] += delta_t[axis];
                advanced = true;
            }
            if !advanced {
                break;
            }
        }
        self.entries_for_indices(entry_indices)
    }

    pub fn nearest_by_center(&self, point: Point3) -> Option<&SpatialEntry<T>> {
        self.entries.iter().min_by(|left, right| {
            distance_squared(left.bounds.center_m(), point)
                .total_cmp(&distance_squared(right.bounds.center_m(), point))
        })
    }

    fn entries_for_indices(&self, entry_indices: BTreeSet<usize>) -> Vec<&SpatialEntry<T>> {
        entry_indices
            .into_iter()
            .filter_map(|index| self.entries.get(index))
            .collect()
    }
}
