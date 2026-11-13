use super::*;
type Link<T> = Option<Arc<Node<T>>>;

struct ExtractedFirstEntry<T> {
    entry: SpatialEntry<T>,
    spatial_key: [u8; 24],
    remaining: Link<T>,
}

#[derive(Debug)]
pub(super) struct Node<T> {
    entry: SpatialEntry<T>,
    spatial_key: [u8; 24],
    bounds: Aabb3,
    height: u32,
    left: Link<T>,
    right: Link<T>,
}

// Ordered IEEE coordinates give deterministic Morton ordering. This is only
// a locality hint: occupied boxes, never these keys, decide candidate pruning.
pub(in crate::quality::spatial_index) fn key(bounds: Aabb3) -> [u8; 24] {
    let coordinates = std::array::from_fn::<_, 3, _>(|axis| {
        let value = bounds.min_m[axis] * 0.5 + bounds.max_m[axis] * 0.5;
        let bits = value.to_bits();
        if bits >> 63 == 0 {
            bits ^ (1 << 63)
        } else {
            !bits
        }
    });
    let mut result = [0; 24];
    for bit in 0..64 {
        for (axis, coordinate) in coordinates.iter().enumerate() {
            let offset = bit * 3 + axis;
            result[offset / 8] |= (((coordinate >> (63 - bit)) & 1) as u8) << (7 - offset % 8);
        }
    }
    result
}

impl<T: Ord + Clone> Node<T> {
    #[cfg(test)]
    pub(super) fn assert_balanced(&self) -> usize {
        let mut count = 1;
        assert_eq!(self.spatial_key, key(self.entry.bounds));
        assert_eq!(self.height, 1 + height(&self.left).max(height(&self.right)));
        assert!(height(&self.left).abs_diff(height(&self.right)) <= 1);
        assert!(
            self.bounds.contains_point(self.entry.bounds.min_m)
                && self.bounds.contains_point(self.entry.bounds.max_m)
        );
        for child in [&self.left, &self.right].into_iter().flatten() {
            assert!(
                self.bounds.contains_point(child.bounds.min_m)
                    && self.bounds.contains_point(child.bounds.max_m)
            );
            count += child.assert_balanced();
        }
        count
    }
    fn new(
        entry: SpatialEntry<T>,
        spatial_key: [u8; 24],
        left: Link<T>,
        right: Link<T>,
    ) -> Arc<Self> {
        let mut bounds = entry.bounds;
        for child in [&left, &right].into_iter().flatten() {
            for axis in 0..3 {
                bounds.min_m[axis] = bounds.min_m[axis].min(child.bounds.min_m[axis]);
                bounds.max_m[axis] = bounds.max_m[axis].max(child.bounds.max_m[axis]);
            }
        }
        Arc::new(Self {
            spatial_key,
            bounds,
            height: 1 + height(&left).max(height(&right)),
            entry,
            left,
            right,
        })
    }

    pub(super) fn build<E>(
        entries: &mut impl Iterator<Item = SpatialEntry<T>>,
        count: usize,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Link<T>, E> {
        checkpoint()?;
        if count == 0 {
            return Ok(None);
        }
        let left = Self::build(entries, count / 2, checkpoint)?;
        let entry = entries.next().expect("ordered spatial inventory count");
        let spatial_key = key(entry.bounds);
        let right = Self::build(entries, count - count / 2 - 1, checkpoint)?;
        Ok(Some(Self::new(entry, spatial_key, left, right)))
    }

    pub(super) fn insert<E>(
        root: &Link<T>,
        entry: SpatialEntry<T>,
        spatial_key: [u8; 24],
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Arc<Self>, E> {
        checkpoint()?;
        let Some(root) = root else {
            return Ok(Self::new(entry, spatial_key, None, None));
        };
        let ordering = (spatial_key, &entry.payload).cmp(&(root.spatial_key, &root.entry.payload));
        let next = match ordering {
            std::cmp::Ordering::Less => Self::new(
                root.entry.clone(),
                root.spatial_key,
                Some(Self::insert(&root.left, entry, spatial_key, checkpoint)?),
                root.right.clone(),
            ),
            std::cmp::Ordering::Greater => Self::new(
                root.entry.clone(),
                root.spatial_key,
                root.left.clone(),
                Some(Self::insert(&root.right, entry, spatial_key, checkpoint)?),
            ),
            std::cmp::Ordering::Equal => {
                Self::new(entry, spatial_key, root.left.clone(), root.right.clone())
            }
        };
        Ok(Self::balance(next))
    }

    pub(super) fn remove<E>(
        root: &Link<T>,
        entry: &SpatialEntry<T>,
        spatial_key: [u8; 24],
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(Link<T>, bool), E> {
        checkpoint()?;
        let Some(root) = root else {
            return Ok((None, false));
        };
        let ordering = (spatial_key, &entry.payload).cmp(&(root.spatial_key, &root.entry.payload));
        let (next, removed) = match ordering {
            std::cmp::Ordering::Less => {
                let (left, removed) = Self::remove(&root.left, entry, spatial_key, checkpoint)?;
                (
                    Self::new(
                        root.entry.clone(),
                        root.spatial_key,
                        left,
                        root.right.clone(),
                    ),
                    removed,
                )
            }
            std::cmp::Ordering::Greater => {
                let (right, removed) = Self::remove(&root.right, entry, spatial_key, checkpoint)?;
                (
                    Self::new(
                        root.entry.clone(),
                        root.spatial_key,
                        root.left.clone(),
                        right,
                    ),
                    removed,
                )
            }
            std::cmp::Ordering::Equal => {
                if root.entry.bounds != entry.bounds {
                    return Ok((Some(root.clone()), false));
                }
                match (&root.left, &root.right) {
                    (None, _) => return Ok((root.right.clone(), true)),
                    (_, None) => return Ok((root.left.clone(), true)),
                    (_, Some(right)) => {
                        let first = Self::take_first(right, checkpoint)?;
                        (
                            Self::new(
                                first.entry,
                                first.spatial_key,
                                root.left.clone(),
                                first.remaining,
                            ),
                            true,
                        )
                    }
                }
            }
        };
        Ok((Some(Self::balance(next)), removed))
    }

    fn take_first<E>(
        root: &Arc<Self>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<ExtractedFirstEntry<T>, E> {
        checkpoint()?;
        let Some(left) = &root.left else {
            return Ok(ExtractedFirstEntry {
                entry: root.entry.clone(),
                spatial_key: root.spatial_key,
                remaining: root.right.clone(),
            });
        };
        let first = Self::take_first(left, checkpoint)?;
        Ok(ExtractedFirstEntry {
            entry: first.entry,
            spatial_key: first.spatial_key,
            remaining: Some(Self::balance(Self::new(
                root.entry.clone(),
                root.spatial_key,
                first.remaining,
                root.right.clone(),
            ))),
        })
    }

    fn balance(root: Arc<Self>) -> Arc<Self> {
        if height(&root.left) > height(&root.right) + 1 {
            let left = root.left.as_ref().expect("left imbalance");
            let root = if height(&left.right) > height(&left.left) {
                Self::new(
                    root.entry.clone(),
                    root.spatial_key,
                    Some(Self::rotate_left(left)),
                    root.right.clone(),
                )
            } else {
                root
            };
            Self::rotate_right(&root)
        } else if height(&root.right) > height(&root.left) + 1 {
            let right = root.right.as_ref().expect("right imbalance");
            let root = if height(&right.left) > height(&right.right) {
                Self::new(
                    root.entry.clone(),
                    root.spatial_key,
                    root.left.clone(),
                    Some(Self::rotate_right(right)),
                )
            } else {
                root
            };
            Self::rotate_left(&root)
        } else {
            root
        }
    }

    fn rotate_left(root: &Arc<Self>) -> Arc<Self> {
        let right = root.right.as_ref().expect("left rotation child");
        Self::new(
            right.entry.clone(),
            right.spatial_key,
            Some(Self::new(
                root.entry.clone(),
                root.spatial_key,
                root.left.clone(),
                right.left.clone(),
            )),
            right.right.clone(),
        )
    }
    fn rotate_right(root: &Arc<Self>) -> Arc<Self> {
        let left = root.left.as_ref().expect("right rotation child");
        Self::new(
            left.entry.clone(),
            left.spatial_key,
            left.left.clone(),
            Some(Self::new(
                root.entry.clone(),
                root.spatial_key,
                left.right.clone(),
                root.right.clone(),
            )),
        )
    }

    pub(super) fn visit<E>(
        root: &Link<T>,
        bounds: Aabb3,
        visit: &mut impl FnMut(&SpatialEntry<T>),
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        checkpoint()?;
        let Some(root) = root else {
            return Ok(());
        };
        if !root.bounds.intersects(bounds) {
            return Ok(());
        }
        if root.entry.bounds.intersects(bounds) {
            visit(&root.entry);
        }
        Self::visit(&root.left, bounds, visit, checkpoint)?;
        Self::visit(&root.right, bounds, visit, checkpoint)
    }
}

fn height<T>(root: &Link<T>) -> u32 {
    root.as_ref().map_or(0, |root| root.height)
}
