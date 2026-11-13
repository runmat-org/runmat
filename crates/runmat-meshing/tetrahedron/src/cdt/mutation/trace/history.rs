//! Chronological replay data shares immutable blocks across bounded search frontiers.
//! Appending or truncating copies at most 128 handles, never edit payloads or
//! the retained prefix. Large cavity inventories remain immutable shared data.
use super::TopologyEdit;
use std::sync::Arc;
const BLOCK_EDITS: usize = 128;

#[derive(Clone, Debug, Default)]
pub(in crate::cdt) struct TopologyEdits {
    blocks: Vec<Arc<Vec<Arc<TopologyEdit>>>>,
    length: usize,
}
impl TopologyEdits {
    pub(super) fn block_count(&self) -> usize {
        self.blocks.len()
    }
    pub(in crate::cdt) fn len(&self) -> usize {
        self.length
    }
    #[cfg(test)]
    pub(in crate::cdt) fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub(in crate::cdt) fn iter(&self) -> impl Iterator<Item = &TopologyEdit> {
        self.blocks
            .iter()
            .flat_map(|block| block.iter().map(Arc::as_ref))
    }
    pub(in crate::cdt) fn starts_with(&self, prefix: &Self) -> bool {
        prefix.length <= self.length && self.iter().zip(prefix.iter()).all(|(a, b)| a == b)
    }
    pub(in crate::cdt::mutation) fn push(&mut self, edit: TopologyEdit) {
        if self.length.is_multiple_of(BLOCK_EDITS) {
            self.blocks.push(Arc::new(Vec::with_capacity(BLOCK_EDITS)));
        }
        Arc::make_mut(self.blocks.last_mut().expect("active edit block")).push(Arc::new(edit));
        self.length += 1;
    }
    pub(in crate::cdt::mutation) fn truncate(&mut self, length: usize) {
        if length >= self.length {
            return;
        }
        self.blocks.truncate(length.div_ceil(BLOCK_EDITS));
        let tail = length % BLOCK_EDITS;
        if tail != 0 {
            Arc::make_mut(self.blocks.last_mut().expect("partial edit block")).truncate(tail);
        }
        self.length = length;
    }
    pub(in crate::cdt::mutation) fn clear(&mut self) {
        self.blocks.clear();
        self.length = 0;
    }
}
#[cfg(test)]
mod tests;
