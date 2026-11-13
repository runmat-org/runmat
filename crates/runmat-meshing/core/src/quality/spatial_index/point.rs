//! Adaptive point broad phase. Immutable branches make append transactional;
//! authoritative predicates and candidate inventory order remain caller-owned.
use super::Aabb3;
use std::sync::Arc;
mod entries;
mod tree;
use entries::Entries;
use tree::Node;

#[derive(Clone, Debug)]
pub struct SpatialPoint<T> {
    pub payload: T,
    position: [f64; 3],
    bounds: Aabb3,
}
impl<T> SpatialPoint<T> {
    pub fn new(position: [f64; 3], payload: T) -> Option<Self> {
        Self::with_bounds(
            position,
            Aabb3 {
                min_m: position,
                max_m: position,
            },
            payload,
        )
    }
    /// Conservatively enclose a constructed point while retaining its routing position.
    pub fn with_bounds(position: [f64; 3], bounds: Aabb3, payload: T) -> Option<Self> {
        ((0..3).all(|axis| {
            position[axis].is_finite()
                && bounds.min_m[axis].is_finite()
                && bounds.max_m[axis].is_finite()
                && bounds.min_m[axis] <= bounds.max_m[axis]
        }) && bounds.contains_point(position))
        .then_some(Self {
            position,
            bounds,
            payload,
        })
    }
    pub fn bounds(&self) -> Aabb3 {
        self.bounds
    }
    pub fn position(&self) -> [f64; 3] {
        self.position
    }
}

#[derive(Clone, Debug)]
pub struct PointSpatialIndex<T> {
    entries: Entries<T>,
    root: Option<Arc<Node>>,
}
impl<T> PointSpatialIndex<T> {
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SpatialPoint<T>> {
        self.entries.iter()
    }
    pub fn from_entries_with_checkpoint<E>(
        entries: Vec<SpatialPoint<T>>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        let entries = Entries::from_points(entries, checkpoint)?;
        let root = Self::build(&entries, checkpoint)?;
        Ok(Self { entries, root })
    }
    fn build<E>(
        entries: &Entries<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Option<Arc<Node>>, E> {
        let Some(first) = entries.get(0) else {
            checkpoint()?;
            return Ok(None);
        };
        let mut bounds = first.bounds;
        let mut indices = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            checkpoint()?;
            for axis in 0..3 {
                bounds.min_m[axis] = bounds.min_m[axis].min(entry.bounds.min_m[axis]);
                bounds.max_m[axis] = bounds.max_m[axis].max(entry.bounds.max_m[axis]);
            }
            indices.push(index);
        }
        Ok(Some(Node::build(
            bounds,
            indices,
            0,
            &|index| entries.get(index).unwrap().position,
            &|index| entries.get(index).unwrap().bounds,
            checkpoint,
        )?))
    }
    pub fn query_bounds_with_checkpoint<E>(
        &self,
        bounds: Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<&SpatialPoint<T>>, E> {
        let mut indices = Vec::new();
        if let Some(root) = &self.root {
            root.query(
                bounds,
                &|index| self.entries.get(index).unwrap().bounds,
                &|_| true,
                &mut |index| indices.push(index),
                checkpoint,
            )?;
        } else {
            checkpoint()?;
        }
        indices.sort_unstable();
        Ok(indices
            .into_iter()
            .map(|index| self.entries.get(index).unwrap())
            .collect())
    }
    /// Stream conservative AABB candidates without a second output inventory or imposed
    /// ordering. Consumers apply their own authoritative inventory order once.
    pub fn visit_bounds_with_checkpoint<E>(
        &self,
        bounds: Aabb3,
        visit: &mut impl FnMut(&SpatialPoint<T>),
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        self.visit_region_with_checkpoint(bounds, &|_| true, visit, checkpoint)
    }
    /// Stream a bounded region. `intersects` must conservatively admit every
    /// occupied box touching the region, including degenerate boxes for points.
    /// The caller retains authority for exact predicates after this broad phase.
    pub fn visit_region_with_checkpoint<E>(
        &self,
        bounds: Aabb3,
        intersects: &impl Fn(Aabb3) -> bool,
        visit: &mut impl FnMut(&SpatialPoint<T>),
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        if let Some(root) = &self.root {
            root.query(
                bounds,
                &|index| self.entries.get(index).unwrap().bounds,
                intersects,
                &mut |index| visit(self.entries.get(index).unwrap()),
                checkpoint,
            )
        } else {
            checkpoint()
        }
    }
}
impl<T: Clone> PointSpatialIndex<T> {
    /// Atomically replace one point while sharing unrelated pages and branches.
    /// Routing outside the retained root follows the same complete rebuild
    /// policy as insertion; interior mesh relocation stays on bounded paths.
    pub fn replace_with_checkpoint<E>(
        &mut self,
        index: usize,
        entry: SpatialPoint<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<bool, E> {
        checkpoint()?;
        if index >= self.entries.len() {
            return Ok(false);
        }
        let retained = self.root.as_ref().expect("nonempty point index").remove(
            index,
            &|slot| self.entries.get(slot).unwrap().position,
            &|slot| self.entries.get(slot).unwrap().bounds,
            checkpoint,
        )?;
        let entries = self.entries.replaced(index, entry, checkpoint)?;
        let root = match retained {
            Some(root)
                if root
                    .region
                    .contains_point(entries.get(index).unwrap().position) =>
            {
                Some(root.insert(
                    index,
                    &|slot| entries.get(slot).unwrap().position,
                    &|slot| entries.get(slot).unwrap().bounds,
                    checkpoint,
                )?)
            }
            _ => Self::build(&entries, checkpoint)?,
        };
        checkpoint()?;
        self.entries = entries;
        self.root = root;
        Ok(true)
    }
    pub fn clone_with_checkpoint<E>(
        &self,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        checkpoint()?;
        Ok(Self {
            entries: self.entries.clone(),
            root: self.root.clone(),
        })
    }
    pub fn insert_with_checkpoint<E>(
        &mut self,
        entry: SpatialPoint<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        let index = self.entries.len();
        let entries = self.entries.appended(entry, checkpoint)?;
        let root = match &self.root {
            Some(root)
                if root
                    .region
                    .contains_point(entries.get(index).unwrap().position) =>
            {
                Some(root.insert(
                    index,
                    &|slot| entries.get(slot).unwrap().position,
                    &|slot| entries.get(slot).unwrap().bounds,
                    checkpoint,
                )?)
            }
            _ => Self::build(&entries, checkpoint)?,
        };
        checkpoint()?;
        self.entries = entries;
        self.root = root;
        Ok(())
    }
}
#[cfg(test)]
mod tests;
