//! Generational allocator for the garbage collector
//!
//! Manages memory allocation across multiple generations, with optimized
//! allocation strategies for different object lifetimes.

use crate::Value;
use crate::{GcConfig, GcError, GcPtr, GcStats, Result};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Size classes for object allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizeClass {
    Small,  // 8-64 bytes
    Medium, // 65-512 bytes
    Large,  // 513+ bytes
}

impl SizeClass {
    pub fn from_size(size: usize) -> Self {
        match size {
            0..=64 => SizeClass::Small,
            65..=512 => SizeClass::Medium,
            _ => SizeClass::Large,
        }
    }

    pub fn allocation_size(&self) -> usize {
        match self {
            SizeClass::Small => 64,
            SizeClass::Medium => 512,
            SizeClass::Large => 4096,
        }
    }
}

/// Represents a generation in the generational heap
#[derive(Debug)]
pub struct Generation {
    /// Generation number (0 = youngest)
    pub number: usize,

    /// Memory blocks for different size classes
    blocks: HashMap<SizeClass, Vec<MemoryBlock>>,

    /// Current allocation position in each size class
    allocation_cursors: HashMap<SizeClass, usize>,

    /// Total bytes allocated in this generation
    allocated_bytes: AtomicUsize,

    /// Maximum size for this generation
    max_size: usize,

    /// Objects that survived collection and may be promoted (stored as addresses)
    survivor_objects: Vec<usize>,

    /// Addresses of allocated objects (object starts) in this generation
    allocated_ptrs: Vec<*const u8>,
}

impl Generation {
    fn new(number: usize, max_size: usize) -> Self {
        let mut blocks = HashMap::new();
        let mut allocation_cursors = HashMap::new();

        // Initialize with one block per size class
        for &size_class in &[SizeClass::Small, SizeClass::Medium, SizeClass::Large] {
            blocks.insert(
                size_class,
                vec![MemoryBlock::new(size_class.allocation_size())],
            );
            allocation_cursors.insert(size_class, 0);
        }

        Self {
            number,
            blocks,
            allocation_cursors,
            allocated_bytes: AtomicUsize::new(0),
            max_size,
            survivor_objects: Vec::new(),
            allocated_ptrs: Vec::new(),
        }
    }

    /// Allocate memory in this generation
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        let size_class = SizeClass::from_size(size);
        let aligned_size = self.align_size(size);

        // Check if we have space
        if self.allocated_bytes.load(Ordering::Relaxed) + aligned_size > self.max_size {
            return Err(GcError::OutOfMemory(format!(
                "Generation {} out of memory",
                self.number
            )));
        }

        let blocks = self.blocks.get_mut(&size_class).unwrap();
        let cursor = self.allocation_cursors.get_mut(&size_class).unwrap();

        // Try to allocate from current block
        if let Some(block) = blocks.get_mut(*cursor) {
            if let Some(ptr) = block.allocate(aligned_size) {
                self.allocated_bytes
                    .fetch_add(aligned_size, Ordering::Relaxed);
                // Track allocation start address for GC bookkeeping
                self.allocated_ptrs.push(ptr as *const u8);
                return Ok(ptr);
            }
        }

        // Need a new block
        let new_block = MemoryBlock::new(std::cmp::max(
            size_class.allocation_size(),
            aligned_size * 2,
        ));
        blocks.push(new_block);
        *cursor = blocks.len() - 1;

        // Allocate from the new block
        let block = blocks.last_mut().unwrap();
        if let Some(ptr) = block.allocate(aligned_size) {
            self.allocated_bytes
                .fetch_add(aligned_size, Ordering::Relaxed);
            self.allocated_ptrs.push(ptr as *const u8);
            Ok(ptr)
        } else {
            Err(GcError::OutOfMemory(
                "Failed to allocate from new block".to_string(),
            ))
        }
    }

    /// Align size to pointer boundary
    fn align_size(&self, size: usize) -> usize {
        (size + std::mem::align_of::<*const u8>() - 1) & !(std::mem::align_of::<*const u8>() - 1)
    }

    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Check if this generation is full
    pub fn is_full(&self) -> bool {
        self.allocated_bytes() >= self.max_size
    }

    /// Mark an object as survivor for potential promotion
    pub fn mark_survivor(&mut self, ptr: *const u8) {
        self.survivor_objects.push(ptr as usize);
    }

    /// Get survivor objects for promotion
    pub fn take_survivors(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.survivor_objects)
    }

    /// Drain allocated object pointers from this generation
    pub fn take_allocated_ptrs(&mut self) -> Vec<*const u8> {
        std::mem::take(&mut self.allocated_ptrs)
    }

    /// Reset generation (after collection)
    pub fn reset(&mut self) {
        for blocks in self.blocks.values_mut() {
            for block in blocks {
                block.reset();
            }
        }
        for cursor in self.allocation_cursors.values_mut() {
            *cursor = 0;
        }
        self.allocated_bytes.store(0, Ordering::Relaxed);
        self.survivor_objects.clear();
        self.allocated_ptrs.clear();
    }
}

/// A block of memory for allocation
#[derive(Debug)]
struct MemoryBlock {
    memory: Vec<u8>,
    current_offset: usize,
    size: usize,
}

impl MemoryBlock {
    fn new(size: usize) -> Self {
        Self {
            memory: vec![0; size],
            current_offset: 0,
            size,
        }
    }

    fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        if self.current_offset + size <= self.size {
            let ptr = unsafe { self.memory.as_mut_ptr().add(self.current_offset) };
            self.current_offset += size;
            Some(ptr)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.current_offset = 0;
        // Zero out memory for security
        self.memory.fill(0);
    }

    fn contains(&self, ptr: *const u8) -> bool {
        let start = self.memory.as_ptr() as usize;
        let end = start + self.size;
        let addr = ptr as usize;
        addr >= start && addr < end
    }
}

/// Main generational allocator
pub struct GenerationalAllocator {
    /// All generations (0 = youngest)
    generations: Vec<Generation>,

    /// Configuration
    config: GcConfig,

    /// Total allocations counter
    total_allocations: AtomicUsize,

    /// Logical promotion tracking (non-moving): pointer -> survival count
    survival_counts: HashMap<*const u8, usize>,

    /// Set of pointers logically promoted to older generation
    promoted_ptrs: HashSet<*const u8>,
}

impl GenerationalAllocator {
    pub fn new(config: &GcConfig) -> Self {
        let mut generations = Vec::new();

        // Create generations with increasing sizes
        let mut gen_size = config.young_generation_size;
        for i in 0..config.num_generations {
            generations.push(Generation::new(i, gen_size));
            gen_size *= 2; // Each generation is twice as large
        }

        Self {
            generations,
            config: config.clone(),
            total_allocations: AtomicUsize::new(0),
            survival_counts: HashMap::new(),
            promoted_ptrs: HashSet::new(),
        }
    }

    /// Allocate a Value object
    pub fn allocate(&mut self, value: Value, stats: &GcStats) -> Result<GcPtr<Value>> {
        let size = self.estimate_value_size(&value);

        // Always allocate in young generation first
        let ptr = self.generations[0].allocate(size)?;

        // Initialize the memory with the value
        unsafe {
            std::ptr::write(ptr as *mut Value, value);
        }

        // Update statistics
        stats.record_allocation(size);

        Ok(unsafe { GcPtr::from_raw(ptr as *const Value) })
    }

    /// Drain allocated object pointers from the young generation
    pub fn young_take_allocations(&mut self) -> Vec<*const u8> {
        self.generations[0].take_allocated_ptrs()
    }

    /// Reset the young generation after a collection cycle
    pub fn young_reset(&mut self) {
        self.generations[0].reset();
    }

    /// Mark a pointer in young generation as survivor (for potential policies)
    pub fn young_mark_survivor(&mut self, ptr: *const u8) {
        self.generations[0].mark_survivor(ptr);
    }

    /// Get count of currently tracked young-generation allocations since last sweep
    pub fn young_allocations_count(&self) -> usize {
        self.generations[0].allocated_ptrs.len()
    }

    /// Promote an object to the next generation
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn promote(&mut self, ptr: *const Value, _from_gen: usize) -> Result<GcPtr<Value>> {
        // Non-moving logical promotion: mark pointer as promoted for barrier/collection logic
        let raw = ptr as *const u8;
        self.promoted_ptrs.insert(raw);
        Ok(unsafe { GcPtr::from_raw(ptr) })
    }

    /// Check if young generation currently tracks any survivors
    pub fn young_has_survivors(&self) -> bool {
        !self.generations[0].survivor_objects.is_empty()
    }

    /// Find which generation contains a pointer
    pub fn find_generation(&self, ptr: *const u8) -> Option<usize> {
        for (i, gen) in self.generations.iter().enumerate() {
            for blocks in gen.blocks.values() {
                for block in blocks {
                    if block.contains(ptr) {
                        return Some(i);
                    }
                }
            }
        }
        None
    }

    /// Get young generation usage as a percentage
    pub fn young_generation_usage(&self) -> f64 {
        let gen = &self.generations[0];
        gen.allocated_bytes() as f64 / gen.max_size as f64
    }

    /// Get total heap usage as a percentage
    pub fn total_usage(&self) -> f64 {
        let total_allocated: usize = self.generations.iter().map(|g| g.allocated_bytes()).sum();
        let total_capacity: usize = self.generations.iter().map(|g| g.max_size).sum();
        total_allocated as f64 / total_capacity as f64
    }

    /// Reconfigure the allocator
    pub fn reconfigure(&mut self, config: &GcConfig) -> Result<()> {
        self.config = config.clone();

        // Resize generations if needed
        if config.num_generations != self.generations.len() {
            return Err(GcError::ConfigError(
                "Cannot change number of generations at runtime".to_string(),
            ));
        }

        // Update generation sizes
        let mut gen_size = config.young_generation_size;
        for (i, gen) in self.generations.iter_mut().enumerate() {
            if gen.max_size != gen_size {
                log::info!(
                    "Resizing generation {} from {} to {} bytes",
                    i,
                    gen.max_size,
                    gen_size
                );
                gen.max_size = gen_size;
            }
            gen_size *= 2; // Each generation is twice as large as the previous
        }

        Ok(())
    }

    /// Increment survival count and promote if threshold reached
    /// Increment survival count; return true if promotion occurred this cycle
    pub fn note_survivor_and_maybe_promote(&mut self, ptr: *const u8) -> bool {
        let count = self.survival_counts.entry(ptr).or_insert(0);
        *count += 1;
        if *count >= self.config.promotion_threshold {
            self.promoted_ptrs.insert(ptr);
            // Decay/reset survival count after promotion to avoid runaway growth
            self.survival_counts.remove(&ptr);
            return true;
        }
        false
    }

    /// Query logical generation of a pointer: 0 for young, >=1 for promoted/old
    pub fn logical_generation(&self, ptr: *const u8) -> Option<usize> {
        if self.promoted_ptrs.contains(&ptr) {
            return Some(1);
        }
        // If pointer belongs to young blocks, treat as gen0; otherwise unknown
        self.find_generation(ptr)
    }

    /// Clear promotion bookkeeping (e.g., after major GC)
    pub fn clear_promotion_state(&mut self) {
        self.survival_counts.clear();
        // Keep promoted set to continue treating them as old
    }

    /// Estimate the memory size of a Value
    #[allow(clippy::only_used_in_recursion)]
    fn estimate_value_size(&self, _value: &Value) -> usize {
        // IMPORTANT: We currently allocate only the outer Value header in the GC heap.
        // Nested payloads (Vecs, strings, tensors) are managed by Rust's allocator and
        // dropped via drop_in_place during sweep. Estimating deep sizes over-reserves and
        // causes artificial OOMs in tests. We'll move to GC-managed aggregates later.
        std::mem::size_of::<Value>()
    }

    /// Get allocator statistics
    pub fn stats(&self) -> AllocatorStats {
        AllocatorStats {
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            generation_usage: self
                .generations
                .iter()
                .map(|g| g.allocated_bytes())
                .collect(),
            young_generation_usage: self.young_generation_usage(),
            total_usage: self.total_usage(),
        }
    }
}

/// Statistics for the allocator
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    pub total_allocations: usize,
    pub generation_usage: Vec<usize>,
    pub young_generation_usage: f64,
    pub total_usage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GcConfig;

    #[test]
    fn test_size_class_classification() {
        assert_eq!(SizeClass::from_size(32), SizeClass::Small);
        assert_eq!(SizeClass::from_size(100), SizeClass::Medium);
        assert_eq!(SizeClass::from_size(1000), SizeClass::Large);
    }

    #[test]
    fn test_generation_allocation() {
        let mut gen = Generation::new(0, 1024);

        // Should be able to allocate
        let ptr = gen.allocate(64).expect("allocation should succeed");
        assert!(!ptr.is_null());
        assert!(gen.allocated_bytes() >= 64);
    }

    #[test]
    fn test_allocator_basic() {
        let config = GcConfig::default();
        let mut allocator = GenerationalAllocator::new(&config);
        let stats = GcStats::new();

        let value = Value::Num(42.0);
        let ptr = allocator
            .allocate(value, &stats)
            .expect("allocation should succeed");

        assert_eq!(*ptr, Value::Num(42.0));
    }
}
