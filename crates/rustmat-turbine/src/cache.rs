//! JIT Function Cache
//!
//! Manages compiled functions, their metadata, and provides efficient lookup
//! for hot code paths in the Turbine JIT compiler.

use crate::CompiledFunction;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cache for compiled JIT functions with proper hit/miss tracking
pub struct FunctionCache {
    functions: HashMap<u64, CompiledFunction>,
    max_size: usize,
    access_counts: HashMap<u64, u64>,
    hit_count: u64,
    miss_count: u64,
    total_requests: u64,
}

impl FunctionCache {
    /// Create a new function cache with default size limit
    pub fn new() -> Self {
        Self::with_capacity(1000) // Default to 1000 compiled functions
    }

    /// Create a cache with specific capacity
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            functions: HashMap::new(),
            max_size,
            access_counts: HashMap::new(),
            hit_count: 0,
            miss_count: 0,
            total_requests: 0,
        }
    }

    /// Insert a compiled function into the cache
    pub fn insert(&mut self, hash: u64, function: CompiledFunction) {
        // If cache is full, evict least recently used function
        if self.functions.len() >= self.max_size {
            self.evict_lru();
        }

        self.functions.insert(hash, function);
        self.access_counts.insert(hash, 0);
    }

    /// Get a compiled function from the cache
    pub fn get(&mut self, hash: u64) -> Option<&CompiledFunction> {
        self.total_requests += 1;

        if let Some(func) = self.functions.get(&hash) {
            // Update access count for LRU tracking
            *self.access_counts.entry(hash).or_insert(0) += 1;
            self.hit_count += 1;
            Some(func)
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Check if a function is cached (doesn't affect hit/miss stats)
    pub fn contains(&self, hash: u64) -> bool {
        self.functions.contains_key(&hash)
    }

    /// Remove a function from the cache
    pub fn remove(&mut self, hash: u64) -> Option<CompiledFunction> {
        self.access_counts.remove(&hash);
        self.functions.remove(&hash)
    }

    /// Clear all cached functions
    pub fn clear(&mut self) {
        self.functions.clear();
        self.access_counts.clear();
        self.hit_count = 0;
        self.miss_count = 0;
        self.total_requests = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.functions.len(),
            capacity: self.max_size,
            hit_rate: self.calculate_hit_rate(),
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            total_requests: self.total_requests,
        }
    }

    /// Evict the least recently used function
    fn evict_lru(&mut self) {
        if let Some((&lru_hash, _)) = self.access_counts.iter().min_by_key(|(_, &count)| count) {
            self.remove(lru_hash);
        }
    }

    fn calculate_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.total_requests as f64
        }
    }
}

impl Default for FunctionCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper for FunctionCache
pub struct ThreadSafeFunctionCache {
    inner: Arc<Mutex<FunctionCache>>,
}

impl ThreadSafeFunctionCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(FunctionCache::new())),
        }
    }

    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(FunctionCache::with_capacity(max_size))),
        }
    }

    pub fn insert(&self, hash: u64, function: CompiledFunction) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.insert(hash, function);
        }
    }

    pub fn get(&self, hash: u64) -> Option<CompiledFunction> {
        if let Ok(mut cache) = self.inner.lock() {
            cache.get(hash).cloned()
        } else {
            None
        }
    }

    pub fn contains(&self, hash: u64) -> bool {
        if let Ok(cache) = self.inner.lock() {
            cache.contains(hash)
        } else {
            false
        }
    }

    pub fn stats(&self) -> Option<CacheStats> {
        if let Ok(cache) = self.inner.lock() {
            Some(cache.stats())
        } else {
            None
        }
    }

    pub fn clear(&self) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.clear();
        }
    }
}

impl Clone for ThreadSafeFunctionCache {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Default for ThreadSafeFunctionCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the function cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub total_requests: u64,
}

impl CacheStats {
    /// Get cache efficiency as a percentage
    pub fn efficiency_percentage(&self) -> f64 {
        self.hit_rate * 100.0
    }

    /// Get cache utilization as a percentage
    pub fn utilization_percentage(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.size as f64 / self.capacity as f64) * 100.0
        }
    }

    /// Check if cache is performing well
    pub fn is_performing_well(&self) -> bool {
        self.hit_rate > 0.8 && self.total_requests > 10
    }
}
