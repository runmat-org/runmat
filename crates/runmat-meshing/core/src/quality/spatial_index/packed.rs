//! Immutable balanced bounds index with contiguous, predictable storage. Use the
//! persistent index when updates/snapshots are required; this owner is bulk-built.
use super::{Aabb3, SpatialEntry};
#[derive(Clone, Copy)]
struct SortRow<T> {
    key: [u8; 24],
    bounds: Aabb3,
    payload: T,
}
#[derive(Clone, Debug)]
struct Node<T> {
    entry: SpatialEntry<T>,
    bounds: Aabb3,
}
#[derive(Debug)]
pub struct PackedBoundsSpatialIndex<T> {
    nodes: Vec<Node<T>>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedBoundsStorageProfile {
    pub retained_bytes: u64,
    /// Temporary rows and cooperative sorting scratch, excluding input entries.
    pub construction_scratch_bytes: u64,
}
impl<T: Copy + Ord> PackedBoundsSpatialIndex<T> {
    pub fn storage_profile(count: usize) -> Option<PackedBoundsStorageProfile> {
        Some(PackedBoundsStorageProfile {
            retained_bytes: u64::try_from(count.checked_mul(std::mem::size_of::<Node<T>>())?)
                .ok()?,
            construction_scratch_bytes: u64::try_from(
                count
                    .checked_mul(std::mem::size_of::<SortRow<T>>())?
                    .checked_mul(2)?,
            )
            .ok()?,
        })
    }
    /// Caller admits the profile before this operation and retains its output
    /// charge for the index lifetime. Input entry storage is separately owned.
    pub fn from_entries_with_checkpoint<E>(
        entries: Vec<SpatialEntry<T>>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Option<Self>, E> {
        checkpoint()?;
        if Self::storage_profile(entries.len()).is_none() {
            return Ok(None);
        }
        let mut rows = Vec::with_capacity(entries.len());
        for entry in entries {
            checkpoint()?;
            if !(0..3).all(|axis| {
                entry.bounds.min_m[axis].is_finite()
                    && entry.bounds.max_m[axis].is_finite()
                    && entry.bounds.min_m[axis] <= entry.bounds.max_m[axis]
            }) {
                return Ok(None);
            }
            rows.push(SortRow {
                key: super::bounds::tree::key(entry.bounds),
                bounds: entry.bounds,
                payload: entry.payload,
            });
        }
        let mut scratch = rows.clone();
        runmat_cpu_execution::sort_copy_with_checkpoint(
            &mut rows,
            &mut scratch,
            |a: &SortRow<T>, b: &SortRow<T>| {
                a.key.cmp(&b.key).then_with(|| a.payload.cmp(&b.payload))
            },
            &mut *checkpoint,
        )?;
        drop(scratch);
        let mut nodes = Vec::with_capacity(rows.len());
        for row in rows {
            checkpoint()?;
            nodes.push(Node {
                entry: SpatialEntry {
                    bounds: row.bounds,
                    payload: row.payload,
                },
                bounds: row.bounds,
            });
        }
        build_bounds(&mut nodes, checkpoint)?;
        Ok(Some(Self { nodes }))
    }
    pub fn visit_bounds_with_checkpoint<E>(
        &self,
        bounds: Aabb3,
        visit: &mut impl FnMut(&SpatialEntry<T>),
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        self.try_visit_bounds_with_checkpoint(
            bounds,
            &mut |entry| {
                visit(entry);
                Ok(())
            },
            checkpoint,
        )
    }
    /// Stream exact caller predicates without allocating a candidate inventory.
    /// A visitor error stops traversal immediately, preserving its original cause.
    pub fn try_visit_bounds_with_checkpoint<E>(
        &self,
        bounds: Aabb3,
        visit: &mut impl FnMut(&SpatialEntry<T>) -> Result<(), E>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        visit_nodes(&self.nodes, bounds, visit, checkpoint)
    }
}
fn build_bounds<T, E>(
    nodes: &mut [Node<T>],
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<Option<Aabb3>, E> {
    checkpoint()?;
    if nodes.is_empty() {
        return Ok(None);
    }
    let middle = nodes.len() / 2;
    let (left, rest) = nodes.split_at_mut(middle);
    let (node, right) = rest.split_first_mut().expect("nonempty subtree");
    for child in [
        build_bounds(left, checkpoint)?,
        build_bounds(right, checkpoint)?,
    ]
    .into_iter()
    .flatten()
    {
        for axis in 0..3 {
            node.bounds.min_m[axis] = node.bounds.min_m[axis].min(child.min_m[axis]);
            node.bounds.max_m[axis] = node.bounds.max_m[axis].max(child.max_m[axis]);
        }
    }
    Ok(Some(node.bounds))
}
fn visit_nodes<T, E>(
    nodes: &[Node<T>],
    bounds: Aabb3,
    visit: &mut impl FnMut(&SpatialEntry<T>) -> Result<(), E>,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<(), E> {
    checkpoint()?;
    if nodes.is_empty() {
        return Ok(());
    }
    let middle = nodes.len() / 2;
    let node = &nodes[middle];
    if !node.bounds.intersects(bounds) {
        return Ok(());
    }
    if node.entry.bounds.intersects(bounds) {
        visit(&node.entry)?;
    }
    visit_nodes(&nodes[..middle], bounds, visit, checkpoint)?;
    visit_nodes(&nodes[middle + 1..], bounds, visit, checkpoint)
}
#[cfg(test)]
mod tests;
