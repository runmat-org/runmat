//! Persistent removal touches one routing path and its occupied leaf.
use super::*;
impl Node {
    pub(in crate::quality::spatial_index::point) fn remove<E>(
        &self,
        index: usize,
        position: &impl Fn(usize) -> [f64; 3],
        entry_bounds: &impl Fn(usize) -> Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Option<Arc<Self>>, E> {
        checkpoint()?;
        match &self.content {
            Content::Leaf(indices) => {
                let mut next = Vec::with_capacity(indices.len().saturating_sub(1));
                for original in indices {
                    checkpoint()?;
                    if *original != index {
                        next.push(*original);
                    }
                }
                if next.is_empty() {
                    Ok(None)
                } else {
                    Self::build(
                        self.region,
                        next,
                        self.depth,
                        position,
                        entry_bounds,
                        checkpoint,
                    )
                    .map(Some)
                }
            }
            Content::Branch { center, children } => {
                let slot = octant(position(index), *center);
                let mut next = children.clone();
                next[slot] = children[slot]
                    .as_ref()
                    .expect("indexed point routing path")
                    .remove(index, position, entry_bounds, checkpoint)?;
                let mut occupied = next.iter().flatten();
                let Some(first) = occupied.next() else {
                    return Ok(None);
                };
                let bounds = occupied.fold(first.bounds, |bounds, child| {
                    include_box(bounds, child.bounds)
                });
                Ok(Some(Arc::new(Self {
                    bounds,
                    region: self.region,
                    depth: self.depth,
                    content: Content::Branch {
                        center: *center,
                        children: next,
                    },
                })))
            }
        }
    }
}
