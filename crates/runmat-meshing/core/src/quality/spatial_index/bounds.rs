//! Persistent balanced bounds broad phase. Updates copy only a branch path;
//! exact geometry and authoritative inventory ordering remain caller-owned.
use super::{Aabb3, SpatialEntry};
use std::{collections::BTreeMap, sync::Arc};
pub(super) mod tree;
use tree::{key, Node};

#[derive(Clone, Debug, Default)]
pub struct BoundsSpatialIndex<T> {
    root: Option<Arc<Node<T>>>,
}

impl<T: Ord + Clone> BoundsSpatialIndex<T> {
    /// Invalid finite/order bounds return None; work rejection returns no index.
    pub fn from_entries_with_checkpoint<E>(
        entries: impl IntoIterator<Item = SpatialEntry<T>>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Option<Self>, E> {
        let mut ordered = BTreeMap::new();
        for entry in entries {
            checkpoint()?;
            if !valid(entry.bounds) {
                return Ok(None);
            }
            ordered.insert((key(entry.bounds), entry.payload.clone()), entry);
        }
        let count = ordered.len();
        let root = Node::build(&mut ordered.into_values(), count, checkpoint)?;
        Ok(Some(Self { root }))
    }

    /// Commit a new root only after all traversal and cancellation checks succeed.
    /// Equal spatial key/payload replaces that entry; changing its key requires
    /// removing the old entry first. Invalid bounds leave the index unchanged.
    pub fn insert_with_checkpoint<E>(
        &mut self,
        entry: SpatialEntry<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<bool, E> {
        if !valid(entry.bounds) {
            return Ok(false);
        }
        let spatial_key = key(entry.bounds);
        let root = Node::insert(&self.root, entry, spatial_key, checkpoint)?;
        checkpoint()?;
        self.root = Some(root);
        Ok(true)
    }

    pub fn remove_with_checkpoint<E>(
        &mut self,
        entry: &SpatialEntry<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<bool, E> {
        let (root, removed) = Node::remove(&self.root, entry, key(entry.bounds), checkpoint)?;
        checkpoint()?;
        self.root = root;
        Ok(removed)
    }

    /// Closed AABB candidates include boundary contact and degenerate boxes.
    /// No tolerance or geometric classification is implied by this query.
    pub fn visit_bounds_with_checkpoint<E>(
        &self,
        bounds: Aabb3,
        visit: &mut impl FnMut(&SpatialEntry<T>),
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        Node::visit(&self.root, bounds, visit, checkpoint)
    }
}

fn valid(bounds: Aabb3) -> bool {
    (0..3).all(|axis| {
        bounds.min_m[axis].is_finite()
            && bounds.max_m[axis].is_finite()
            && bounds.min_m[axis] <= bounds.max_m[axis]
    })
}

#[cfg(test)]
mod tests;
