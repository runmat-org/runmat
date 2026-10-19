//! Work-accounted grid construction, append and conservative bounds queries.
use super::*;

impl<T> UniformGridSpatialIndex<T> {
    pub fn from_entries_with_checkpoint<E>(
        entries: Vec<SpatialEntry<T>>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut bounds = entries.first().map(|entry| entry.bounds).unwrap_or(Aabb3 {
            min_m: [0.0; 3],
            max_m: [0.0; 3],
        });
        for entry in &entries {
            checkpoint()?;
            for axis in 0..3 {
                bounds.min_m[axis] = bounds.min_m[axis].min(entry.bounds.min_m[axis]);
                bounds.max_m[axis] = bounds.max_m[axis].max(entry.bounds.max_m[axis]);
            }
        }
        let dimensions = grid_dimensions(bounds, entries.len());
        let mut cells = BTreeMap::<[usize; 3], Vec<usize>>::new();
        for (entry_index, entry) in entries.iter().enumerate() {
            let min_cell = cell_for_point(bounds, dimensions, entry.bounds.min_m);
            let max_cell = cell_for_point(bounds, dimensions, entry.bounds.max_m);
            for x in min_cell[0]..=max_cell[0] {
                for y in min_cell[1]..=max_cell[1] {
                    for z in min_cell[2]..=max_cell[2] {
                        checkpoint()?;
                        cells.entry([x, y, z]).or_default().push(entry_index);
                    }
                }
            }
        }
        Ok(Self {
            entries,
            bounds,
            dimensions,
            cells,
        })
    }

    pub fn query_bounds_with_checkpoint<E>(
        &self,
        bounds: Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<&SpatialEntry<T>>, E> {
        checkpoint()?;
        if self.entries.is_empty() || !self.bounds.intersects(bounds) {
            return Ok(Vec::new());
        }
        let min_cell = cell_for_point(self.bounds, self.dimensions, bounds.min_m);
        let max_cell = cell_for_point(self.bounds, self.dimensions, bounds.max_m);
        let mut entry_indices = BTreeSet::new();
        for x in min_cell[0]..=max_cell[0] {
            for y in min_cell[1]..=max_cell[1] {
                for z in min_cell[2]..=max_cell[2] {
                    checkpoint()?;
                    if let Some(indices) = self.cells.get(&[x, y, z]) {
                        for index in indices {
                            checkpoint()?;
                            entry_indices.insert(*index);
                        }
                    }
                }
            }
        }
        let mut result = Vec::new();
        for index in entry_indices {
            checkpoint()?;
            if let Some(entry) = self
                .entries
                .get(index)
                .filter(|entry| entry.bounds.intersects(bounds))
            {
                result.push(entry);
            }
        }
        Ok(result)
    }
}

impl<T: Clone> UniformGridSpatialIndex<T> {
    /// Copy actual retained entries and cell memberships cooperatively. This is
    /// used only when a mutable trial must detach from shared immutable storage.
    pub fn clone_with_checkpoint<E>(
        &self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut entries = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            checkpoint()?;
            entries.push(entry.clone());
        }
        let mut cells = BTreeMap::new();
        for (key, indices) in &self.cells {
            checkpoint()?;
            let mut copied = Vec::with_capacity(indices.len());
            for index in indices {
                checkpoint()?;
                copied.push(*index);
            }
            cells.insert(*key, copied);
        }
        Ok(Self {
            entries,
            bounds: self.bounds,
            dimensions: self.dimensions,
            cells,
        })
    }

    /// Append without rebuilding an unchanged grid. Out-of-bounds growth is
    /// prepared separately so cancellation preserves the original inventory.
    pub fn insert_with_checkpoint<E>(
        &mut self,
        entry: SpatialEntry<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        checkpoint()?;
        if self.entries.is_empty()
            || !self.bounds.contains_point(entry.bounds.min_m)
            || !self.bounds.contains_point(entry.bounds.max_m)
            || grid_dimensions(self.bounds, self.entries.len() + 1) != self.dimensions
        {
            let mut entries = Vec::with_capacity(self.entries.len() + 1);
            for existing in &self.entries {
                checkpoint()?;
                entries.push(existing.clone());
            }
            entries.push(entry);
            *self = Self::from_entries_with_checkpoint(entries, checkpoint)?;
            return Ok(());
        }
        let min_cell = cell_for_point(self.bounds, self.dimensions, entry.bounds.min_m);
        let max_cell = cell_for_point(self.bounds, self.dimensions, entry.bounds.max_m);
        let mut keys = Vec::new();
        for x in min_cell[0]..=max_cell[0] {
            for y in min_cell[1]..=max_cell[1] {
                for z in min_cell[2]..=max_cell[2] {
                    checkpoint()?;
                    keys.push([x, y, z]);
                }
            }
        }
        let index = self.entries.len();
        self.entries.push(entry);
        for key in keys {
            self.cells.entry(key).or_default().push(index);
        }
        Ok(())
    }
}
