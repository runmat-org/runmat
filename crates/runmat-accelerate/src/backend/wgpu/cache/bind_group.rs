use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
    inner: Mutex<HashMap<BindGroupKey, Arc<wgpu::BindGroup>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl BindGroupCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn get_or_create<F>(
        &self,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
        create: F,
    ) -> Arc<wgpu::BindGroup>
    where
        F: FnOnce() -> Arc<wgpu::BindGroup>,
    {
        match Self::make_key(layout, entries) {
            Some(key) => {
                if let Some(existing) = self
                    .inner
                    .lock()
                    .ok()
                    .and_then(|map| map.get(&key).cloned())
                {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    log::debug!("bind_group_cache hit layout_ptr={:#x}", key.layout_ptr);
                    return existing;
                }
                let bind_group = create();
                let layout_ptr = key.layout_ptr;
                if let Ok(mut map) = self.inner.lock() {
                    map.insert(key, bind_group.clone());
                }
                self.misses.fetch_add(1, Ordering::Relaxed);
                log::debug!("bind_group_cache miss layout_ptr={:#x}", layout_ptr);
                bind_group
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                create()
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
    }

    fn make_key(
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
    ) -> Option<BindGroupKey> {
        let mut bindings: SmallVec<[BindingKey; 8]> = SmallVec::with_capacity(entries.len());
        for entry in entries {
            match &entry.resource {
                wgpu::BindingResource::Buffer(buffer_binding) => {
                    let buffer_ptr = buffer_binding.buffer.global_id().inner() as usize;
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
                _ => {
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
