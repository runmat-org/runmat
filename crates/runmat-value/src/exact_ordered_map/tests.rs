use super::*;
use crate::{ExactHostAllocationPermit, ExactHostAllocationReservation};
use std::{
    hash::BuildHasher,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[derive(Debug)]
struct Permit {
    limit: usize,
    live: Arc<AtomicUsize>,
}
#[derive(Debug)]
struct Claim {
    bytes: usize,
    live: Arc<AtomicUsize>,
}
impl ExactHostAllocationReservation for Claim {}
impl Drop for Claim {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
impl ExactHostAllocationPermit for Permit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.live
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit)
            })
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(Arc::new(Claim {
            bytes,
            live: self.live.clone(),
        }))
    }
}
fn key(name: &str, permit: &Permit) -> Utf8Text {
    Utf8Text::try_copy_str_exact_with(name, permit, |_| Ok::<(), std::convert::Infallible>(()))
        .unwrap()
}
fn permit(limit: usize) -> Permit {
    Permit {
        limit,
        live: Arc::new(AtomicUsize::new(0)),
    }
}

#[test]
fn replacement_duplicate_and_shift_removal_preserve_order_and_map_equality() {
    let permit = permit(usize::MAX);
    let mut first = ExactOrderedMapBuilder::try_new(3, &permit).unwrap();
    first.try_insert_prepared(key("beta", &permit), 2).unwrap();
    first.try_insert_prepared(key("alpha", &permit), 1).unwrap();
    assert_eq!(
        first.try_insert_prepared(key("beta", &permit), 3).unwrap(),
        Some(2)
    );
    let duplicate = first
        .try_insert_new_prepared(key("alpha", &permit), 9)
        .unwrap_err();
    let (kind, rejected_key, rejected_value) = duplicate.into_parts();
    assert_eq!(kind, MapInsertErrorKind::DuplicateKey);
    assert_eq!(rejected_key.as_str(), "alpha");
    assert_eq!(rejected_value, 9);
    let first = first.finish();
    assert_eq!(
        first.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        ["beta", "alpha"]
    );
    assert_eq!(first.get("Beta"), None);
    let mut second = ExactOrderedMapBuilder::try_new(3, &permit).unwrap();
    second
        .try_insert_prepared(key("alpha", &permit), 1)
        .unwrap();
    second
        .try_insert_prepared(key("discard", &permit), 0)
        .unwrap();
    second.try_insert_prepared(key("beta", &permit), 3).unwrap();
    assert_eq!(second.remove("discard"), Some(0));
    let second = second.finish();
    assert_eq!(
        second.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert_eq!(first, second);
}

#[test]
fn exact_quotes_refusal_partial_drop_and_final_alias_retirement() {
    let quote = ExactOrderedMap::<u64>::quote(2).unwrap();
    assert_eq!(quote.retained_claims, 2);
    let denied = permit(quote.requested_bytes - 1);
    assert!(matches!(
        ExactOrderedMapBuilder::<u64>::try_new(2, &denied),
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    ));
    assert_eq!(denied.live.load(Ordering::SeqCst), 0);
    assert!(ExactOrderedMap::<u64>::quote(usize::MAX).is_err());

    let key_bytes = ExactOrderedMap::<u64>::key_quote("key").unwrap().0;
    let permit = permit(quote.requested_bytes + key_bytes);
    let mut partial = ExactOrderedMapBuilder::try_new(2, &permit).unwrap();
    let rejected = partial
        .try_insert_prepared(Utf8Text::new("unclaimed".into()), 11_u64)
        .unwrap_err();
    assert_eq!(rejected.into_parts().0, MapInsertErrorKind::KeyNotAdmitted);
    let mut spare_empty = String::with_capacity(8);
    spare_empty.clear();
    assert_eq!(
        partial
            .try_insert_prepared(Utf8Text::new(spare_empty), 12_u64)
            .unwrap_err()
            .into_parts()
            .0,
        MapInsertErrorKind::KeyNotAdmitted
    );
    partial
        .try_insert_prepared(Utf8Text::new(String::new()), 13_u64)
        .unwrap();
    partial
        .try_insert_prepared(key("key", &permit), 7_u64)
        .unwrap();
    assert_eq!(
        permit.live.load(Ordering::SeqCst),
        quote.requested_bytes + key_bytes
    );
    drop(partial);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);

    let mut builder = ExactOrderedMapBuilder::try_new(2, &permit).unwrap();
    builder
        .try_insert_prepared(key("key", &permit), 7_u64)
        .unwrap();
    let map = builder.finish();
    let alias = map.clone();
    assert!(std::ptr::eq(
        map.entries().as_ptr(),
        alias.entries().as_ptr()
    ));
    drop(map);
    assert_eq!(alias.get("key"), Some(&7));
    assert_eq!(
        permit.live.load(Ordering::SeqCst),
        quote.requested_bytes + key_bytes
    );
    drop(alias);
    assert_eq!(permit.live.load(Ordering::SeqCst), 0);
}

#[test]
fn fixed_collision_chain_survives_removal_and_capacity_refusal() {
    let permit = permit(usize::MAX);
    let mut builder = ExactOrderedMapBuilder::try_new(4, &permit).unwrap();
    let mask = builder.index.len() - 1;
    // Four times the bucket count guarantees a four-key collision group,
    // independent of the randomized seed; no probabilistically bounded loop.
    let mut buckets = vec![Vec::new(); mask + 1];
    for n in 0..4 * buckets.len() {
        let name = format!("key-{n}");
        let bucket = builder.hasher.hash_one(name.as_str()) as usize & mask;
        buckets[bucket].push(name);
    }
    let names = buckets
        .into_iter()
        .find(|bucket| bucket.len() >= 4)
        .unwrap()
        .into_iter()
        .take(4)
        .collect::<Vec<_>>();
    for (value, name) in names.iter().enumerate() {
        builder
            .try_insert_prepared(key(name, &permit), value)
            .unwrap();
    }
    let full = builder
        .try_insert_prepared(key("extra", &permit), 99)
        .unwrap_err();
    assert_eq!(full.into_parts().0, MapInsertErrorKind::CapacityExceeded);
    assert_eq!(builder.remove(&names[1]), Some(1));
    let map = builder.finish();
    for value in [0, 2, 3] {
        assert_eq!(map.get(&names[value]), Some(&value));
    }
    assert_eq!(map.get(&names[1]), None);
}
