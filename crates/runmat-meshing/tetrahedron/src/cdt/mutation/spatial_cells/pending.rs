//! First-write undo for deferred index changes; checkpoint never copies the map.
use super::*;
use std::ops::Deref;
type Change = (Option<Aabb3>, Option<Aabb3>);
type Journal = BTreeMap<CellKey, Option<Change>>;

#[derive(Clone, Debug, Default)]
pub(super) struct PendingChanges {
    entries: BTreeMap<CellKey, Change>,
    journals: Vec<Journal>,
}
impl PendingChanges {
    pub(super) fn begin(&mut self) -> usize {
        self.journals.push(BTreeMap::new());
        self.journals.len()
    }
    pub(super) fn commit(&mut self, depth: usize) {
        assert_eq!(
            self.journals.len(),
            depth,
            "cell cache checkpoints are nested"
        );
        self.journals.pop().expect("live pending checkpoint");
    }
    pub(super) fn rollback(&mut self, depth: usize) {
        assert_eq!(
            self.journals.len(),
            depth,
            "cell cache checkpoints are nested"
        );
        let journal = self.journals.pop().expect("live pending checkpoint");
        for (key, original) in journal {
            self.set(key, original);
        }
    }
    pub(super) fn set(&mut self, key: CellKey, value: Option<Change>) {
        for journal in &mut self.journals {
            journal
                .entry(key)
                .or_insert_with(|| self.entries.get(&key).copied());
        }
        match value {
            Some(value) => {
                self.entries.insert(key, value);
            }
            None => {
                self.entries.remove(&key);
            }
        }
        for journal in &mut self.journals {
            if journal
                .get(&key)
                .is_some_and(|old| change_equal(*old, self.entries.get(&key).copied()))
            {
                journal.remove(&key);
            }
        }
    }
    pub(super) fn remove_first(&mut self) {
        let key = *self
            .entries
            .first_key_value()
            .expect("nonempty pending map")
            .0;
        self.set(key, None);
    }
    #[cfg(test)]
    pub(super) fn undo_entries(&self) -> usize {
        self.journals.iter().map(BTreeMap::len).sum()
    }
}
impl Deref for PendingChanges {
    type Target = BTreeMap<CellKey, Change>;
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}
fn change_equal(left: Option<Change>, right: Option<Change>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some((a, b)), Some((c, d))) => bounds_equal(a, c) && bounds_equal(b, d),
        _ => false,
    }
}
pub(super) fn bounds_equal(left: Option<Aabb3>, right: Option<Aabb3>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => a
            .min_m
            .into_iter()
            .chain(a.max_m)
            .zip(b.min_m.into_iter().chain(b.max_m))
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        _ => false,
    }
}
#[cfg(test)]
mod tests;
