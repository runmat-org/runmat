//! JIT Function Cache
//!
//! Manages compiled functions, their metadata, and provides efficient lookup
//! for hot code paths in the Turbine JIT compiler.

use std::collections::HashMap;
use crate::CompiledFunction;

/// Cache for compiled JIT functions
pub struct FunctionCache {
    functions: HashMap<u64, CompiledFunction>,
    max_size: usize,
    access_counts: HashMap<u64, u64>,
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
        if let Some(func) = self.functions.get(&hash) {
            // Update access count for LRU tracking
            *self.access_counts.entry(hash).or_insert(0) += 1;
            Some(func)
        } else {
            None
        }
    }
    
    /// Check if a function is cached
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
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.functions.len(),
            capacity: self.max_size,
            hit_rate: self.calculate_hit_rate(),
        }
    }
    
    /// Evict the least recently used function
    fn evict_lru(&mut self) {
        if let Some((&lru_hash, _)) = self.access_counts.iter()
            .min_by_key(|(_, &count)| count) {
            self.remove(lru_hash);
        }
    }
    
    fn calculate_hit_rate(&self) -> f64 {
        let total_accesses: u64 = self.access_counts.values().sum();
        if total_accesses == 0 {
            0.0
        } else {
            let cache_hits = self.access_counts.len() as u64;
            cache_hits as f64 / total_accesses as f64
        }
    }
}

impl Default for FunctionCache {
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
} 