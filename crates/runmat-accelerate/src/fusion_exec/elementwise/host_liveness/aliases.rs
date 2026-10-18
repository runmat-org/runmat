//! Canonical ValueId equivalence classes for allocation-free alias operators.

use crate::graph::ValueId;

pub(super) struct Aliases {
    ids: Vec<ValueId>,
    parent: Vec<usize>,
}

impl Aliases {
    pub(super) fn new(ids: Vec<ValueId>) -> Self {
        let mut parent = Vec::with_capacity(ids.len());
        parent.extend(0..ids.len());
        Self { parent, ids }
    }

    pub(super) fn len(&self) -> usize {
        self.ids.len()
    }

    fn position(&self, id: ValueId) -> Option<usize> {
        self.ids.binary_search(&id).ok()
    }

    fn root(&self, mut index: usize) -> usize {
        while self.parent[index] != index {
            index = self.parent[index];
        }
        index
    }

    pub(super) fn root_index(&self, id: ValueId) -> Option<usize> {
        self.position(id).map(|index| self.root(index))
    }

    pub(super) fn union(&mut self, left: ValueId, right: ValueId) -> Option<()> {
        let left = self.root(self.position(left)?);
        let right = self.root(self.position(right)?);
        if left != right {
            let (root, child) = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            self.parent[child] = root;
        }
        Some(())
    }
}
