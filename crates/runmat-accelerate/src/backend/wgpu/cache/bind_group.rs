use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

mod entries;
use entries::BindGroupEntries;

/// Invalidation could not establish completed bookkeeping. Neither variant
/// asserts that a GPU resource or device has failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheInvalidationError {
    Poisoned,
    BookkeepingCapacity,
}

impl std::fmt::Display for CacheInvalidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Poisoned => "bind-group cache bookkeeping is poisoned",
            Self::BookkeepingCapacity => "bind-group invalidation bookkeeping capacity unavailable",
        })
    }
}

impl std::error::Error for CacheInvalidationError {}

#[derive(Clone, Debug, Eq)]
struct BindingKey {
    binding: u32,
    buffer_ptr: usize,
    offset: u64,
    size: Option<u64>,
}

impl PartialEq for BindingKey {
    fn eq(&self, other: &Self) -> bool {
        self.binding == other.binding
            && self.buffer_ptr == other.buffer_ptr
            && self.offset == other.offset
            && self.size == other.size
    }
}

impl Hash for BindingKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.binding.hash(state);
        self.buffer_ptr.hash(state);
        self.offset.hash(state);
        self.size.hash(state);
    }
}

#[derive(Clone, Debug, Eq)]
struct BindGroupKey {
    layout_ptr: usize,
    bindings: SmallVec<[BindingKey; 8]>,
}

impl PartialEq for BindGroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.layout_ptr == other.layout_ptr && self.bindings == other.bindings
    }
}

impl Hash for BindGroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layout_ptr.hash(state);
        self.bindings.hash(state);
    }
}

pub struct BindGroupCache {
    inner: Mutex<BindGroupEntries<Arc<wgpu::BindGroup>>>,
    hits: AtomicU64,
    misses: AtomicU64,
    per_layout: Mutex<HashMap<usize, (u64, u64)>>,
}

impl Drop for BindGroupCache {
    fn drop(&mut self) {
        let entries = std::mem::take(self.inner.get_mut().unwrap_or_else(|e| e.into_inner()));
        for group in entries.groups.into_values() {
            if runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind(|| drop(group))
                .is_err()
            {
                let _ = runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind(|| {
                    log::error!(
                        "WGPU bind-group terminal reference release unwound; no retry attempted"
                    );
                });
            }
        }
    }
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BindGroupCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BindGroupEntries::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            per_layout: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_create<F>(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
        create: F,
    ) -> anyhow::Result<Arc<wgpu::BindGroup>>
    where
        F: FnOnce(&runmat_accelerate_api::WgpuEmission<'_>) -> Arc<wgpu::BindGroup>,
    {
        match Self::make_key(layout, entries) {
            Some(key) => {
                if let Some(existing) = self
                    .inner
                    .lock()
                    .ok()
                    .and_then(|entries| entries.groups.get(&key).cloned())
                {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    log::debug!("bind_group_cache hit layout_ptr={:#x}", key.layout_ptr);
                    if let Ok(mut per) = self.per_layout.lock() {
                        let entry = per.entry(key.layout_ptr).or_insert((0, 0));
                        entry.0 = entry.0.saturating_add(1);
                    }
                    return Ok(existing);
                }
                super::scope::allow_creation(emission.mode())?;
                let bind_group = create(emission);
                let layout_ptr = key.layout_ptr;
                let replaced = self
                    .inner
                    .lock()
                    .ok()
                    .and_then(|mut entries| entries.insert(key, bind_group.clone()));
                drop(replaced);
                self.misses.fetch_add(1, Ordering::Relaxed);
                log::debug!("bind_group_cache miss layout_ptr={:#x}", layout_ptr);
                if let Ok(mut per) = self.per_layout.lock() {
                    let entry = per.entry(layout_ptr).or_insert((0, 0));
                    entry.1 = entry.1.saturating_add(1);
                }
                Ok(bind_group)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                // Unkeyed resources are returned to the transaction owner, not
                // published in this cache before detached scope resolution.
                Ok(create(emission))
            }
        }
    }

    pub fn counters(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    pub fn reset_counters(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        if let Ok(mut per) = self.per_layout.lock() {
            per.clear();
        }
    }

    pub fn per_layout_counters(&self) -> HashMap<usize, (u64, u64)> {
        self.per_layout
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default()
    }

    pub fn invalidate_buffer(&self, buffer_ptr: usize) -> Result<(), CacheInvalidationError> {
        let removed = remove_cached_buffer(&self.inner, buffer_ptr)?;
        drop(removed);
        Ok(())
    }

    #[cfg(all(test, panic = "unwind"))]
    pub(crate) fn poison_for_retirement_test(&self) {
        let _guard = self.inner.lock().unwrap();
        panic!("injected bind-group cache poison");
    }

    fn make_key(
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
    ) -> Option<BindGroupKey> {
        let mut bindings: SmallVec<[BindingKey; 8]> = SmallVec::with_capacity(entries.len());
        for entry in entries {
            match &entry.resource {
                wgpu::BindingResource::Buffer(buffer_binding) => {
                    let buffer_ptr = buffer_binding.buffer as *const wgpu::Buffer as usize;
                    let size = buffer_binding.size.map(|v| v.get());
                    log::debug!(
                        "bind_group_cache key binding={} ptr={:#x} offset={} size={:?}",
                        entry.binding,
                        buffer_ptr,
                        buffer_binding.offset,
                        size
                    );
                    bindings.push(BindingKey {
                        binding: entry.binding,
                        buffer_ptr,
                        offset: buffer_binding.offset,
                        size,
                    });
                }
                other => {
                    log::debug!(
                        "bind_group_cache unsupported resource at binding {}: {:?}",
                        entry.binding,
                        other
                    );
                    return None;
                }
            }
        }
        Some(BindGroupKey {
            layout_ptr: layout as *const _ as usize,
            bindings,
        })
    }
}

fn remove_cached_buffer<T>(
    entries: &Mutex<BindGroupEntries<T>>,
    buffer_ptr: usize,
) -> Result<Vec<T>, CacheInvalidationError> {
    // Both tables are unchanged on poison/capacity failure. Resource drops
    // happen in the caller, after this guard has gone out of scope.
    let mut entries = entries
        .lock()
        .map_err(|_| CacheInvalidationError::Poisoned)?;
    entries.remove_buffer(buffer_ptr)
}
