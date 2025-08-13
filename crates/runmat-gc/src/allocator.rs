//! Generational allocator for the garbage collector
//!
//! Manages memory allocation across multiple generations, with optimized
//! allocation strategies for different object lifetimes.

use crate::{GcConfig, GcError, GcPtr, GcStats, Result};
use runmat_builtins::Value;
use std::collections::HashMap;
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

    /// Promote an object to the next generation
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn promote(&mut self, ptr: *const Value, from_gen: usize) -> Result<GcPtr<Value>> {
        if from_gen + 1 >= self.generations.len() {
            // Already in oldest generation
            return Ok(unsafe { GcPtr::from_raw(ptr) });
        }

        // Copy object to next generation
        let value = unsafe { (*ptr).clone() };
        let size = self.estimate_value_size(&value);

        let new_ptr = self.generations[from_gen + 1].allocate(size)?;
        unsafe {
            std::ptr::write(new_ptr as *mut Value, value);
        }

        Ok(unsafe { GcPtr::from_raw(new_ptr as *const Value) })
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

    /// Estimate the memory size of a Value
    #[allow(clippy::only_used_in_recursion)]
    fn estimate_value_size(&self, value: &Value) -> usize {
        match value {
            Value::Int(_) | Value::Num(_) | Value::Bool(_) => std::mem::size_of::<Value>(),
            Value::String(s) => std::mem::size_of::<Value>() + s.len(),
            Value::Tensor(m) => {
                std::mem::size_of::<Value>()
                    + std::mem::size_of::<runmat_builtins::Tensor>()
                    + m.data.len() * std::mem::size_of::<f64>()
            }
            Value::Cell(cells) => {
                std::mem::size_of::<Value>()
                    + cells.data.len() * std::mem::size_of::<Value>()
                    + cells
                        .data
                        .iter()
                        .map(|v| self.estimate_value_size(v))
                        .sum::<usize>()
            }
            Value::GpuTensor(_h) => {
                // Handle is small; device memory is managed externally by backend
                std::mem::size_of::<Value>() + std::mem::size_of::<runmat_accelerate_api::GpuTensorHandle>()
            }
            Value::Object(obj) => {
                std::mem::size_of::<Value>()
                    + obj.class_name.len()
                    + obj.properties.iter().map(|(k, v)| k.len() + self.estimate_value_size(v)).sum::<usize>()
            }
            Value::FunctionHandle(name) => std::mem::size_of::<Value>() + name.len(),
            Value::ClassRef(name) => std::mem::size_of::<Value>() + name.len(),
            Value::Closure(c) => std::mem::size_of::<Value>() + c.function_name.len() + c.captures.iter().map(|v| self.estimate_value_size(v)).sum::<usize>(),
            Value::MException(e) => std::mem::size_of::<Value>() + e.identifier.len() + e.message.len() + e.stack.iter().map(|s| s.len()).sum::<usize>(),
        }
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
