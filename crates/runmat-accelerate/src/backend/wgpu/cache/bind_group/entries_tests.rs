use super::super::{remove_cached_buffer, BindingKey};
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn key(layout: usize, buffers: &[usize]) -> BindGroupKey {
    BindGroupKey {
        layout_ptr: layout,
        bindings: buffers
            .iter()
            .enumerate()
            .map(|(binding, pointer)| BindingKey {
                binding: binding as u32,
                buffer_ptr: *pointer,
                offset: 0,
                size: Some(16),
            })
            .collect(),
    }
}

#[test]
fn invalidation_updates_both_tables_and_preserves_unrelated_groups() {
    let entries = Mutex::new(BindGroupEntries::default());
    let shared = key(1, &[10, 20, 10]);
    let other = key(1, &[20, 30]);
    {
        let mut entries = entries.lock().unwrap();
        assert_eq!(entries.insert(shared.clone(), 1), None);
        assert_eq!(entries.insert(other.clone(), 2), None);
        assert_eq!(entries.insert(shared.clone(), 3), Some(1));
    }
    assert_eq!(remove_cached_buffer(&entries, 10).unwrap(), vec![3]);
    {
        let entries = entries.lock().unwrap();
        assert!(!entries.groups.contains_key(&shared));
        assert_eq!(entries.groups.get(&other), Some(&2));
        assert!(!entries.by_buffer.contains_key(&10));
        assert_eq!(entries.by_buffer[&20].len(), 1);
        assert!(entries.by_buffer[&20].contains(&other));
    }
    assert!(remove_cached_buffer(&entries, 10).unwrap().is_empty());
    assert_eq!(remove_cached_buffer(&entries, 30).unwrap(), vec![2]);
    let entries = entries.lock().unwrap();
    assert!(entries.groups.is_empty());
    assert!(entries.by_buffer.is_empty());
}

#[cfg(panic = "unwind")]
#[test]
fn poison_keeps_group_and_reverse_evidence_without_partial_invalidation() {
    let entries = Mutex::new(BindGroupEntries::default());
    let registered = key(1, &[10, 20]);
    assert!(entries
        .lock()
        .unwrap()
        .insert(registered.clone(), 7)
        .is_none());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = entries.lock().unwrap();
        panic!("poison cache bookkeeping");
    }))
    .is_err());
    assert_eq!(
        remove_cached_buffer(&entries, 10),
        Err(CacheInvalidationError::Poisoned)
    );
    assert_eq!(
        remove_cached_buffer(&entries, 20),
        Err(CacheInvalidationError::Poisoned)
    );
    let state = match entries.lock() {
        Ok(_) => panic!("cache must remain quarantined after poison"),
        Err(error) => error.into_inner(),
    };
    assert_eq!(state.groups.get(&registered), Some(&7));
    assert!(state.by_buffer[&10].contains(&registered));
    assert!(state.by_buffer[&20].contains(&registered));
}

struct ReentrantDrop(Box<dyn Fn() + Send + Sync>);
impl Drop for ReentrantDrop {
    fn drop(&mut self) {
        (self.0)();
    }
}

#[test]
fn invalidated_and_replaced_resources_drop_after_cache_unlock() {
    let entries = Arc::new(Mutex::new(BindGroupEntries::default()));
    let drops = Arc::new(AtomicUsize::new(0));
    let resource = || {
        let entries = Arc::downgrade(&entries);
        let drops = drops.clone();
        ReentrantDrop(Box::new(move || {
            let entries = entries.upgrade().unwrap();
            assert!(
                entries.try_lock().is_ok(),
                "resource destructor reentered cache lock"
            );
            drops.fetch_add(1, Ordering::Relaxed);
        }))
    };
    let registered = key(1, &[10]);
    let replaced = entries
        .lock()
        .unwrap()
        .insert(registered.clone(), resource());
    assert!(replaced.is_none());
    let replaced = entries.lock().unwrap().insert(registered, resource());
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(replaced);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    let removed = remove_cached_buffer(&entries, 10).unwrap();
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    drop(removed);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}
