use std::{collections::hash_map::RandomState, hash::BuildHasher};

use super::OrderedMapEntry;
use crate::ExactHostNumericAllocationError;

/// Zero means empty; occupied slots store entry ordinal plus one.
#[derive(Clone, Copy, Debug)]
pub(super) struct IndexSlot(pub(super) usize);

pub(super) enum Lookup {
    Found { entry: usize },
    Vacant { bucket: usize },
}

pub(super) fn index_len(capacity: usize) -> Result<usize, ExactHostNumericAllocationError> {
    if capacity == 0 {
        return Ok(0);
    }
    capacity
        .checked_mul(2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
}

pub(super) fn lookup<V>(
    entries: &[OrderedMapEntry<V>],
    index: &[IndexSlot],
    hasher: &RandomState,
    name: &str,
) -> Option<Lookup> {
    if index.is_empty() {
        return None;
    }
    let mask = index.len() - 1;
    let mut bucket = hasher.hash_one(name) as usize & mask;
    for _ in 0..index.len() {
        let ordinal = index[bucket].0;
        if ordinal == 0 {
            return Some(Lookup::Vacant { bucket });
        }
        let entry = ordinal - 1;
        if entries[entry].key.as_str() == name {
            return Some(Lookup::Found { entry });
        }
        bucket = bucket.wrapping_add(1) & mask;
    }
    None
}

pub(super) fn rebuild<V>(
    entries: &[OrderedMapEntry<V>],
    index: &mut [IndexSlot],
    hasher: &RandomState,
) {
    index.fill(IndexSlot(0));
    for entry in 0..entries.len() {
        let Some(Lookup::Vacant { bucket }) = lookup(
            &entries[..entry],
            index,
            hasher,
            entries[entry].key.as_str(),
        ) else {
            unreachable!("unique map keys and a half-full fixed index leave a vacant bucket");
        };
        index[bucket] = IndexSlot(entry + 1);
    }
}
