use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
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
    index: Mutex<HashMap<usize, HashSet<BindGroupKey>>>,
    hits: AtomicU64,
    misses: AtomicU64,
    per_layout: Mutex<HashMap<usize, (u64, u64)>>,
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BindGroupCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            index: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            per_layout: Mutex::new(HashMap::new()),
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
                    if let Ok(mut per) = self.per_layout.lock() {
                        let entry = per.entry(key.layout_ptr).or_insert((0, 0));
                        entry.0 = entry.0.saturating_add(1);
                    }
                    return existing;
                }
                let bind_group = create();
                let layout_ptr = key.layout_ptr;
                let key_clone = key.clone();
                if let Ok(mut map) = self.inner.lock() {
                    map.insert(key, bind_group.clone());
                }
                if let Ok(mut index) = self.index.lock() {
                    for binding in &key_clone.bindings {
                        index
                            .entry(binding.buffer_ptr)
                            .or_insert_with(HashSet::new)
                            .insert(key_clone.clone());
                    }
                }
                self.misses.fetch_add(1, Ordering::Relaxed);
                log::debug!("bind_group_cache miss layout_ptr={:#x}", layout_ptr);
                if let Ok(mut per) = self.per_layout.lock() {
                    let entry = per.entry(layout_ptr).or_insert((0, 0));
                    entry.1 = entry.1.saturating_add(1);
                }
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

    pub fn invalidate_buffer(&self, buffer_ptr: usize) {
        let removed_keys = {
            let mut keys_to_remove: Vec<BindGroupKey> = Vec::new();
            if let Ok(mut index) = self.index.lock() {
                if let Some(keys_set) = index.remove(&buffer_ptr) {
                    keys_to_remove.extend(keys_set);
                    let mut empty_ptrs: HashSet<usize> = HashSet::new();
                    for key in &keys_to_remove {
                        for binding in &key.bindings {
                            if binding.buffer_ptr == buffer_ptr {
                                continue;
                            }
                            if let Some(set) = index.get_mut(&binding.buffer_ptr) {
                                set.remove(key);
                                if set.is_empty() {
                                    empty_ptrs.insert(binding.buffer_ptr);
                                }
                            }
                        }
                    }
                    for ptr in empty_ptrs {
                        index.remove(&ptr);
                    }
                } else {
                    return;
                }
            } else {
                return;
            }
            keys_to_remove
        };

        if let Ok(mut map) = self.inner.lock() {
            for key in removed_keys {
                map.remove(&key);
            }
        }
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
