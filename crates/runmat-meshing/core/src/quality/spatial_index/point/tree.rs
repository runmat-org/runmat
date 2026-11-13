use super::{Aabb3, Arc};
mod removal;
const LEAF_POINTS: usize = 32;
const MAX_DEPTH: u8 = 52;
#[derive(Debug)]
pub(super) struct Node {
    pub bounds: Aabb3,
    pub(super) region: Aabb3,
    depth: u8,
    content: Content,
}
#[derive(Debug)]
enum Content {
    Leaf(Vec<usize>),
    Branch {
        center: [f64; 3],
        children: Box<[Option<Arc<Node>>; 8]>,
    },
}
impl Node {
    pub fn build<E>(
        region: Aabb3,
        indices: Vec<usize>,
        depth: u8,
        position: &impl Fn(usize) -> [f64; 3],
        entry_bounds: &impl Fn(usize) -> Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Arc<Self>, E> {
        checkpoint()?;
        // Tight occupied bounds prune empty space; the partition region remains
        // stable so future appends anywhere in that region are routed correctly.
        let first = position(indices[0]);
        let mut routing_bounds = Aabb3 {
            min_m: first,
            max_m: first,
        };
        let mut bounds = entry_bounds(indices[0]);
        for index in &indices {
            checkpoint()?;
            routing_bounds = include(routing_bounds, position(*index));
            bounds = include_box(bounds, entry_bounds(*index));
        }
        let center = center(routing_bounds);
        let can_split = (0..3).any(|axis| {
            center[axis] > routing_bounds.min_m[axis] && center[axis] < routing_bounds.max_m[axis]
        });
        let content = if indices.len() <= LEAF_POINTS || depth == MAX_DEPTH || !can_split {
            Content::Leaf(indices)
        } else {
            let mut groups: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
            for index in indices {
                checkpoint()?;
                groups[octant(position(index), center)].push(index);
            }
            let mut children = std::array::from_fn(|_| None);
            for (slot, group) in groups.into_iter().enumerate() {
                if !group.is_empty() {
                    children[slot] = Some(Self::build(
                        child_bounds(region, center, slot),
                        group,
                        depth + 1,
                        position,
                        entry_bounds,
                        checkpoint,
                    )?);
                }
            }
            Content::Branch {
                center,
                children: Box::new(children),
            }
        };
        Ok(Arc::new(Self {
            bounds,
            region,
            depth,
            content,
        }))
    }
    pub fn insert<E>(
        &self,
        index: usize,
        position: &impl Fn(usize) -> [f64; 3],
        entry_bounds: &impl Fn(usize) -> Aabb3,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Arc<Self>, E> {
        checkpoint()?;
        let content = match &self.content {
            Content::Leaf(indices) => {
                let mut next = Vec::with_capacity(indices.len() + 1);
                for original in indices {
                    checkpoint()?;
                    next.push(*original);
                }
                next.push(index);
                return Self::build(
                    self.region,
                    next,
                    self.depth,
                    position,
                    entry_bounds,
                    checkpoint,
                );
            }
            Content::Branch { center, children } => {
                let center = *center;
                let slot = octant(position(index), center);
                let mut next = children.clone();
                next[slot] = Some(match &children[slot] {
                    Some(child) => child.insert(index, position, entry_bounds, checkpoint)?,
                    None => Self::build(
                        child_bounds(self.region, center, slot),
                        vec![index],
                        self.depth + 1,
                        position,
                        entry_bounds,
                        checkpoint,
                    )?,
                });
                Content::Branch {
                    center,
                    children: next,
                }
            }
        };
        Ok(Arc::new(Self {
            bounds: include_box(self.bounds, entry_bounds(index)),
            region: self.region,
            depth: self.depth,
            content,
        }))
    }
    pub fn query<E>(
        &self,
        bounds: Aabb3,
        entry_bounds: &impl Fn(usize) -> Aabb3,
        intersects: &impl Fn(Aabb3) -> bool,
        visit: &mut impl FnMut(usize),
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        checkpoint()?;
        if !self.bounds.intersects(bounds) || !intersects(self.bounds) {
            return Ok(());
        }
        match &self.content {
            Content::Leaf(points) => {
                for index in points {
                    checkpoint()?;
                    let occupied = entry_bounds(*index);
                    if bounds.intersects(occupied) && intersects(occupied) {
                        visit(*index);
                    }
                }
            }
            Content::Branch { children, .. } => {
                for child in children.iter().flatten() {
                    child.query(bounds, entry_bounds, intersects, visit, checkpoint)?;
                }
            }
        }
        Ok(())
    }
}
fn include(bounds: Aabb3, point: [f64; 3]) -> Aabb3 {
    Aabb3 {
        min_m: std::array::from_fn(|axis| bounds.min_m[axis].min(point[axis])),
        max_m: std::array::from_fn(|axis| bounds.max_m[axis].max(point[axis])),
    }
}
fn center(bounds: Aabb3) -> [f64; 3] {
    std::array::from_fn(|axis| 0.5 * bounds.min_m[axis] + 0.5 * bounds.max_m[axis])
}
fn octant(point: [f64; 3], center: [f64; 3]) -> usize {
    (0..3).fold(0, |slot, axis| {
        slot | (usize::from(point[axis] >= center[axis]) << axis)
    })
}
fn child_bounds(bounds: Aabb3, center: [f64; 3], slot: usize) -> Aabb3 {
    Aabb3 {
        min_m: std::array::from_fn(|axis| {
            if slot & (1 << axis) == 0 {
                bounds.min_m[axis]
            } else {
                center[axis]
            }
        }),
        max_m: std::array::from_fn(|axis| {
            if slot & (1 << axis) == 0 {
                center[axis]
            } else {
                bounds.max_m[axis]
            }
        }),
    }
}

fn include_box(bounds: Aabb3, entry: Aabb3) -> Aabb3 {
    include(include(bounds, entry.min_m), entry.max_m)
}
