//! Append-only point pages. Trials share unchanged pages and copy one bounded path.
use super::{Arc, SpatialPoint};
const PAGE_BITS: u32 = 5;
const PAGE_SIZE: usize = 1 << PAGE_BITS;

#[derive(Debug)]
pub(super) enum Page<T> {
    Leaf(Vec<SpatialPoint<T>>),
    Branch(Box<[Option<Arc<Page<T>>>; PAGE_SIZE]>),
}

#[derive(Debug)]
pub(super) struct Entries<T> {
    pub(super) root: Option<Arc<Page<T>>>,
    shift: u32,
    len: usize,
}

impl<T> Clone for Entries<T> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            shift: self.shift,
            len: self.len,
        }
    }
}

impl<T> Entries<T> {
    pub fn from_points<E>(
        points: Vec<SpatialPoint<T>>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        let len = points.len();
        let mut points = points.into_iter();
        let mut pages = Vec::new();
        loop {
            let mut leaf = Vec::with_capacity(PAGE_SIZE);
            for point in points.by_ref().take(PAGE_SIZE) {
                checkpoint()?;
                leaf.push(point);
            }
            if leaf.is_empty() {
                break;
            }
            pages.push(Arc::new(Page::Leaf(leaf)));
        }
        let mut shift = 0;
        while pages.len() > 1 {
            let mut previous = pages.into_iter();
            let mut next = Vec::new();
            loop {
                let mut children = std::array::from_fn(|_| None);
                let mut occupied = false;
                for (slot, page) in previous.by_ref().take(PAGE_SIZE).enumerate() {
                    checkpoint()?;
                    children[slot] = Some(page);
                    occupied = true;
                }
                if !occupied {
                    break;
                }
                next.push(Arc::new(Page::Branch(Box::new(children))));
            }
            pages = next;
            shift += PAGE_BITS;
        }
        checkpoint()?;
        Ok(Self {
            root: pages.pop(),
            shift,
            len,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn get(&self, index: usize) -> Option<&SpatialPoint<T>> {
        if index >= self.len {
            return None;
        }
        let mut page = self.root.as_deref()?;
        let mut shift = self.shift;
        loop {
            match page {
                Page::Leaf(points) => return points.get(index & (PAGE_SIZE - 1)),
                Page::Branch(children) => {
                    page = children[(index >> shift) & (PAGE_SIZE - 1)].as_deref()?;
                    shift -= PAGE_BITS;
                }
            }
        }
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SpatialPoint<T>> {
        (0..self.len).map(|index| self.get(index).expect("complete immutable point inventory"))
    }
}

impl<T: Clone> Entries<T> {
    pub fn replaced<E>(
        &self,
        index: usize,
        point: SpatialPoint<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        assert!(
            index < self.len,
            "replacement belongs to retained point inventory"
        );
        let root = Some(replace(
            self.root.as_deref().expect("nonempty point pages"),
            self.shift,
            index,
            point,
            checkpoint,
        )?);
        Ok(Self {
            root,
            shift: self.shift,
            len: self.len,
        })
    }
    pub fn appended<E>(
        &self,
        point: SpatialPoint<T>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        // An allocated inventory cannot approach usize::MAX; still fence arithmetic.
        let len = self
            .len
            .checked_add(1)
            .expect("point inventory index capacity");
        let mut shift = self.shift;
        let mut root = self.root.clone();
        if root.is_some() && shift + PAGE_BITS < usize::BITS && self.len >> (shift + PAGE_BITS) != 0
        {
            checkpoint()?;
            let mut children = std::array::from_fn(|_| None);
            children[0] = root;
            root = Some(Arc::new(Page::Branch(Box::new(children))));
            shift += PAGE_BITS;
        }
        let root = Some(append(root.as_deref(), shift, self.len, point, checkpoint)?);
        Ok(Self { root, shift, len })
    }
}

fn replace<T: Clone, E>(
    page: &Page<T>,
    shift: u32,
    index: usize,
    point: SpatialPoint<T>,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<Arc<Page<T>>, E> {
    checkpoint()?;
    match page {
        Page::Leaf(original) => {
            let mut points = Vec::with_capacity(original.len());
            for (slot, entry) in original.iter().enumerate() {
                checkpoint()?;
                points.push(if slot == (index & (PAGE_SIZE - 1)) {
                    point.clone()
                } else {
                    entry.clone()
                });
            }
            Ok(Arc::new(Page::Leaf(points)))
        }
        Page::Branch(original) => {
            let mut children = std::array::from_fn(|slot| original[slot].clone());
            let slot = (index >> shift) & (PAGE_SIZE - 1);
            children[slot] = Some(replace(
                children[slot].as_deref().expect("occupied point path"),
                shift - PAGE_BITS,
                index,
                point,
                checkpoint,
            )?);
            Ok(Arc::new(Page::Branch(Box::new(children))))
        }
    }
}

fn append<T: Clone, E>(
    page: Option<&Page<T>>,
    shift: u32,
    index: usize,
    point: SpatialPoint<T>,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<Arc<Page<T>>, E> {
    checkpoint()?;
    if shift == 0 {
        let mut points = Vec::with_capacity(PAGE_SIZE);
        if let Some(Page::Leaf(original)) = page {
            for entry in original {
                checkpoint()?;
                points.push(entry.clone());
            }
        }
        points.push(point);
        return Ok(Arc::new(Page::Leaf(points)));
    }
    let mut children = std::array::from_fn(|_| None);
    if let Some(Page::Branch(original)) = page {
        for (slot, child) in original.iter().enumerate() {
            checkpoint()?;
            children[slot] = child.clone();
        }
    }
    let slot = (index >> shift) & (PAGE_SIZE - 1);
    children[slot] = Some(append(
        children[slot].as_deref(),
        shift - PAGE_BITS,
        index,
        point,
        checkpoint,
    )?);
    Ok(Arc::new(Page::Branch(Box::new(children))))
}
