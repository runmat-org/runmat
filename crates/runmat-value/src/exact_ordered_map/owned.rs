use super::{ExactOrderedMap, OrderedMapEntry};
use crate::{ExactArrayIntoIter, ExactArrayStorage, Utf8Text};

/// Consumed exact entries keep their original physical claim until the
/// backing is freed, including early iterator drop. Key claims remain with
/// each moved key. No child or key bytes are cloned by unique consumption.
#[derive(Debug)]
pub struct OwnedOrderedMapEntries<V> {
    entries: ExactArrayIntoIter<OrderedMapEntry<V>>,
}
impl<V> Iterator for OwnedOrderedMapEntries<V> {
    type Item = (Utf8Text, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|entry| (entry.key, entry.value))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}
impl<V> ExactSizeIterator for OwnedOrderedMapEntries<V> {}

impl<V> ExactOrderedMap<V> {
    pub fn try_into_entries(self) -> Result<OwnedOrderedMapEntries<V>, Self> {
        let Self {
            entries,
            index,
            hasher,
        } = self;
        match entries.try_into_owned() {
            Ok(entries) => Ok(OwnedOrderedMapEntries {
                entries: entries.into_iter(),
            }),
            Err(entries) => Err(Self {
                entries,
                index,
                hasher,
            }),
        }
    }
}
impl<V: Clone> ExactOrderedMap<V> {
    /// Explicit ordinary-policy copying boundary. Finite consumers must use
    /// unique consumption or construct an independently admitted destination.
    pub fn into_entries_unbounded(self) -> OwnedOrderedMapEntries<V> {
        match self.try_into_entries() {
            Ok(entries) => entries,
            Err(source) => {
                let entries = ExactArrayStorage::unbounded(source.entries.as_slice().to_vec());
                let entries = match entries.try_into_owned() {
                    Ok(entries) => entries,
                    Err(_) => unreachable!("fresh ordinary entries are unique"),
                };
                OwnedOrderedMapEntries {
                    entries: entries.into_iter(),
                }
            }
        }
    }
}
