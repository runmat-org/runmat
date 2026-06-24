//! Generational allocator for the garbage collector
//!
//! Manages memory allocation across multiple generations, with optimized
//! allocation strategies for different object lifetimes.

use crate::Value;
use crate::{GcConfig, GcError, GcHandle, GcStats, Result};
use std::collections::{HashMap, HashSet};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
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

    /// Objects that survived collection and may be promoted.
    survivor_objects: Vec<*const u8>,

    /// Addresses of allocated objects (object starts) in this generation
    allocated_ptrs: Vec<*const u8>,

    /// Value-sized slots reclaimed by collection and available for reuse.
    free_value_slots: Vec<*mut u8>,
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
            free_value_slots: Vec::new(),
        }
    }

    /// Allocate memory in this generation
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        let size_class = SizeClass::from_size(size);
        let aligned_size = self.align_size(size);

        if aligned_size <= std::mem::size_of::<Value>() {
            if let Some(ptr) = self.free_value_slots.pop() {
                self.allocated_bytes
                    .fetch_add(aligned_size, Ordering::Relaxed);
                self.allocated_ptrs.push(ptr as *const u8);
                return Ok(ptr);
            }
        }

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

    /// Align size to the type stored by this allocator.
    fn align_size(&self, size: usize) -> usize {
        let align = std::mem::align_of::<Value>();
        (size + align - 1) & !(align - 1)
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
        self.survivor_objects.push(ptr);
    }

    /// Get survivor objects for promotion
    pub fn take_survivors(&mut self) -> Vec<*const u8> {
        std::mem::take(&mut self.survivor_objects)
    }

    /// Drain allocated object pointers from this generation
    pub fn take_allocated_ptrs(&mut self) -> Vec<*const u8> {
        std::mem::take(&mut self.allocated_ptrs)
    }

    fn note_value_slot_freed(&mut self, ptr: *const u8) {
        if self.find_block_containing(ptr).is_some() {
            let value_size = std::mem::size_of::<Value>();
            self.allocated_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_sub(value_size))
                })
                .ok();
            self.free_value_slots.push(ptr as *mut u8);
        }
    }

    fn take_tracked_ptrs_for_reset(&mut self) -> Vec<*const u8> {
        let mut ptrs = std::mem::take(&mut self.allocated_ptrs);
        ptrs.extend(std::mem::take(&mut self.survivor_objects));
        ptrs
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
        self.free_value_slots.clear();
    }

    fn find_block_containing(&self, ptr: *const u8) -> Option<&MemoryBlock> {
        self.blocks
            .values()
            .flat_map(|blocks| blocks.iter())
            .find(|block| block.contains(ptr))
    }
}

/// A block of memory for allocation
#[derive(Debug)]
struct MemoryBlock {
    memory: Vec<MaybeUninit<Value>>,
    current_slot: usize,
    size_bytes: usize,
}

impl MemoryBlock {
    fn new(size_bytes: usize) -> Self {
        let slot_count = Self::slots_for_bytes(size_bytes);
        let mut memory = Vec::with_capacity(slot_count);
        memory.resize_with(slot_count, MaybeUninit::uninit);
        Self {
            memory,
            current_slot: 0,
            size_bytes: slot_count * std::mem::size_of::<Value>(),
        }
    }

    fn slots_for_bytes(size_bytes: usize) -> usize {
        let value_size = std::mem::size_of::<Value>();
        std::cmp::max(1, size_bytes.div_ceil(value_size))
    }

    fn allocate(&mut self, size_bytes: usize) -> Option<*mut u8> {
        let slots_needed = Self::slots_for_bytes(size_bytes);
        if self.current_slot + slots_needed <= self.memory.len() {
            // Use the vector's raw buffer pointer directly. Indexing would
            // create a temporary `&mut` to an element, and later allocations
            // can invalidate earlier raw tags under Stacked Borrows.
            let ptr = unsafe { self.memory.as_mut_ptr().add(self.current_slot) }.cast::<u8>();
            self.current_slot += slots_needed;
            Some(ptr)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.current_slot = 0;
    }

    fn contains(&self, ptr: *const u8) -> bool {
        let start = self.memory.as_ptr() as usize;
        let end = start + self.size_bytes;
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

    /// Pointers whose `Value` payload is currently initialized and live.
    live_ptrs: HashSet<*const u8>,

    /// Allocation epoch for each live pointer. Incremented before each new
    /// allocation so stale handles to reused slots do not validate.
    live_epochs: HashMap<*const u8, usize>,

    /// Monotonic allocation epoch source.
    next_epoch: usize,
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
            live_ptrs: HashSet::new(),
            live_epochs: HashMap::new(),
            next_epoch: 1,
        }
    }

    /// Allocate a Value object
    pub fn allocate(&mut self, value: Value, stats: &GcStats) -> Result<GcHandle> {
        let size = self.estimate_value_size(&value);
        let epoch = self.next_epoch;
        let next_epoch = epoch.checked_add(1).ok_or_else(|| {
            GcError::OutOfMemory("GC allocation epoch space exhausted".to_string())
        })?;

        // Always allocate in young generation first
        let ptr = self.generations[0].allocate(size)?;

        // Initialize the memory with the value
        unsafe {
            std::ptr::write(ptr as *mut Value, value);
        }
        let raw = ptr.cast_const();
        self.next_epoch = next_epoch;
        self.live_ptrs.insert(raw);
        self.live_epochs.insert(raw, epoch);

        // Update statistics
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        stats.record_allocation(size);

        value_handle_from_ptr(ptr.cast::<Value>(), epoch)
    }

    /// Drain young-generation pointers that must be considered during the next
    /// collection. This includes new allocations and survivors from earlier
    /// minor collections.
    pub fn young_take_collection_candidates(&mut self) -> Vec<*const u8> {
        let mut seen = HashSet::new();
        self.generations[0]
            .take_tracked_ptrs_for_reset()
            .into_iter()
            .filter(|ptr| self.live_ptrs.contains(ptr) && seen.insert(*ptr))
            .collect()
    }

    /// Return all currently live Value pointers for a major collection sweep.
    pub fn all_live_collection_candidates(&self) -> Vec<*const u8> {
        self.live_ptrs.iter().copied().collect()
    }

    /// Drain all pointers the allocator still believes may contain initialized
    /// Values. This is only for whole-GC teardown/reset paths.
    pub fn take_tracked_value_ptrs_for_reset(&mut self) -> Vec<(*const Value, Option<GcHandle>)> {
        let mut ptrs = Vec::new();
        let mut seen = HashSet::new();

        for generation in &mut self.generations {
            for ptr in generation.take_tracked_ptrs_for_reset() {
                if seen.insert(ptr) {
                    ptrs.push(ptr);
                }
            }
        }

        for ptr in self.survival_counts.keys().copied() {
            if seen.insert(ptr) {
                ptrs.push(ptr);
            }
        }

        let promoted_ptrs: Vec<*const u8> = self.promoted_ptrs.drain().collect();
        for ptr in promoted_ptrs {
            if seen.insert(ptr) {
                ptrs.push(ptr);
            }
        }

        let ptrs: Vec<(*const Value, Option<GcHandle>)> = ptrs
            .into_iter()
            .filter(|ptr| self.live_ptrs.contains(ptr))
            .map(|ptr| (ptr.cast::<Value>(), self.handle_for_live_ptr(ptr)))
            .collect();

        self.survival_counts.clear();
        self.live_ptrs.clear();
        self.live_epochs.clear();
        ptrs
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
    pub fn promote(&mut self, ptr: *const Value, _from_gen: usize) -> Result<GcHandle> {
        // Non-moving logical promotion: mark pointer as promoted for barrier/collection logic
        let raw = ptr as *const u8;
        let Some(epoch) = self.live_epochs.get(&raw).copied() else {
            return Err(GcError::InvalidPointer(format!(
                "cannot promote non-live GC value pointer {:p}",
                ptr
            )));
        };
        self.promoted_ptrs.insert(raw);
        value_handle_from_ptr(ptr, epoch)
    }

    /// Mark an initialized Value slot as no longer live after sweep/reset drops
    /// it. Later validation must reject handles to this slot until it is reused
    /// by a fresh allocation.
    pub fn note_value_dropped(&mut self, ptr: *const u8) -> bool {
        if !self.live_ptrs.remove(&ptr) {
            return false;
        }
        self.live_epochs.remove(&ptr);
        self.survival_counts.remove(&ptr);
        self.promoted_ptrs.remove(&ptr);
        if let Some(generation) = self.generations.get_mut(0) {
            generation.note_value_slot_freed(ptr);
        }
        true
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

    /// Return whether a pointer names a currently initialized GC Value.
    pub fn is_live_value_ptr(&self, ptr: *const Value) -> bool {
        self.live_ptrs.contains(&ptr.cast::<u8>())
    }

    pub fn is_live_handle(&self, handle: &GcHandle) -> bool {
        let ptr = handle.addr() as *const u8;
        self.live_epochs
            .get(&ptr)
            .is_some_and(|epoch| *epoch == handle.epoch())
    }

    pub fn handle_for_live_ptr(&self, ptr: *const u8) -> Option<GcHandle> {
        let epoch = self.live_epochs.get(&ptr).copied()?;
        value_handle_from_ptr(ptr.cast::<Value>(), epoch).ok()
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

fn value_handle_from_ptr(ptr: *const Value, epoch: usize) -> Result<GcHandle> {
    let raw = NonNull::new(ptr as *mut ())
        .ok_or_else(|| GcError::InvalidPointer("null GC allocation pointer".to_string()))?;
    // SAFETY: allocator callers only pass addresses returned by generation
    // allocation or promotion of existing generation-owned Value slots, paired
    // with the allocator-issued live allocation epoch for that slot.
    Ok(unsafe { GcHandle::from_parts_unchecked(raw, epoch) })
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

        // SAFETY: this handle was returned by the allocator under test and no
        // collection or mutation runs before this read.
        assert_eq!(
            unsafe { &*ptr.as_ptr_unchecked().cast::<Value>().as_ptr() },
            &Value::Num(42.0)
        );
    }

    #[test]
    fn allocator_returns_value_aligned_pointers() {
        let config = GcConfig::default();
        let mut allocator = GenerationalAllocator::new(&config);
        let stats = GcStats::new();
        let align = std::mem::align_of::<Value>();

        for i in 0..128 {
            let ptr = allocator
                .allocate(Value::Num(i as f64), &stats)
                .expect("allocation should succeed");
            let addr = ptr.addr();
            assert_eq!(addr % align, 0, "GC allocation was not Value-aligned");
        }
    }

    #[test]
    fn young_generation_reuses_dropped_value_slots() {
        let config = GcConfig {
            young_generation_size: std::mem::size_of::<Value>() * 2,
            ..GcConfig::default()
        };
        let mut allocator = GenerationalAllocator::new(&config);
        let stats = GcStats::new();

        let first = allocator
            .allocate(Value::String("first".to_string()), &stats)
            .expect("first allocation");
        let _survivor = allocator
            .allocate(Value::String("survivor".to_string()), &stats)
            .expect("survivor allocation");

        let first_ptr = first.addr() as *const u8;
        allocator.note_value_dropped(first_ptr);
        unsafe {
            std::ptr::drop_in_place(first_ptr as *mut Value);
        }

        let reused = allocator
            .allocate(Value::String("reused".to_string()), &stats)
            .expect("dropped young slot should be reusable");
        assert_eq!(reused.addr(), first.addr());
        assert_ne!(reused.epoch(), first.epoch());
        assert!(!allocator.is_live_handle(&first));
        assert!(allocator.is_live_handle(&reused));
    }
}
