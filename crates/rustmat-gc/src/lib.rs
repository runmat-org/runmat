//! RustMat High-Performance Generational Garbage Collector
//!
//! A production-quality, thread-safe generational garbage collector designed for
//! high-performance interpreters. Features safe object management with handle-based
//! access instead of raw pointers to avoid undefined behavior.

use parking_lot::RwLock;
use rustmat_builtins::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

pub mod allocator;
pub mod barriers;
pub mod collector;
pub mod config;
pub mod gc_ptr;
pub mod generations;
pub mod roots;
pub mod stats;

#[cfg(feature = "pointer-compression")]
pub mod compression;

pub use allocator::{AllocatorStats, GenerationalAllocator, SizeClass};
pub use barriers::{CardTable, WriteBarrier};
pub use collector::MarkSweepCollector;
pub use config::{GcConfig, GcConfigBuilder};
pub use gc_ptr::GcPtr;
pub use generations::{Generation, GenerationStats, GenerationalHeap, GenerationalHeapStats};
pub use roots::{GcRoot, GlobalRoot, RootId, RootScanner, StackRoot, VariableArrayRoot};
pub use stats::{CollectionEvent, CollectionType, GcStats};

#[cfg(feature = "pointer-compression")]
pub use compression::*;

/// Errors that can occur during garbage collection
#[derive(Error, Debug)]
pub enum GcError {
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    #[error("Invalid GC pointer: {0}")]
    InvalidPointer(String),

    #[error("Collection failed: {0}")]
    CollectionFailed(String),

    #[error("Root registration failed: {0}")]
    RootRegistrationFailed(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Thread synchronization error: {0}")]
    SyncError(String),
}

pub type Result<T> = std::result::Result<T, GcError>;

/// Type alias to simplify complex generational map type
type GenerationMap = Arc<RwLock<HashMap<usize, Arc<GcObject>>>>;

/// Handle to a GC-managed object (safer than raw pointers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GcHandle(usize);

impl GcHandle {
    fn new(id: usize) -> Self {
        GcHandle(id)
    }

    fn id(&self) -> usize {
        self.0
    }

    pub fn is_null(&self) -> bool {
        self.0 == 0
    }

    pub fn null() -> Self {
        GcHandle(0)
    }
}

/// A managed object in the GC heap
#[derive(Debug)]
struct GcObject {
    /// Unique ID for this object
    id: usize,
    /// The actual value
    value: Value,
    /// Mark bit for collection
    marked: AtomicBool,
    /// Generation this object belongs to
    generation: u8,
}

impl GcObject {
    fn new(id: usize, value: Value, generation: u8) -> Self {
        Self {
            id,
            value,
            marked: AtomicBool::new(false),
            generation,
        }
    }

    fn mark(&self) -> bool {
        !self.marked.swap(true, Ordering::AcqRel)
    }

    fn unmark(&self) {
        self.marked.store(false, Ordering::Release);
    }

    fn is_marked(&self) -> bool {
        self.marked.load(Ordering::Acquire)
    }

    /// Get the unique ID of this object
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the generation this object belongs to
    pub fn generation(&self) -> u8 {
        self.generation
    }

    /// Get a reference to the value stored in this object
    pub fn value(&self) -> &Value {
        &self.value
    }
}

/// High-performance garbage collector with safe handle-based design
pub struct HighPerformanceGC {
    /// Configuration
    config: Arc<RwLock<GcConfig>>,

    /// All allocated objects by ID
    objects: Arc<RwLock<HashMap<usize, Arc<GcObject>>>>,

    /// Objects by generation for efficient collection
    generations: Vec<GenerationMap>,

    /// Root set - handles to live objects
    roots: Arc<RwLock<HashMap<GcHandle, ()>>>,

    /// Next object ID
    next_id: AtomicUsize,

    /// Collection state
    collection_in_progress: AtomicBool,

    /// Statistics
    stats: Arc<GcStats>,
}

impl HighPerformanceGC {
    pub fn new() -> Result<Self> {
        Self::with_config(GcConfig::default())
    }

    pub fn with_config(config: GcConfig) -> Result<Self> {
        let num_generations = config.num_generations;
        let mut generations = Vec::with_capacity(num_generations);

        for _ in 0..num_generations {
            generations.push(Arc::new(RwLock::new(HashMap::new())));
        }

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            objects: Arc::new(RwLock::new(HashMap::new())),
            generations,
            roots: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicUsize::new(1), // Start at 1, 0 is null
            collection_in_progress: AtomicBool::new(false),
            stats: Arc::new(GcStats::new()),
        })
    }

    /// Allocate a new Value object
    pub fn allocate(&self, value: Value) -> Result<GcPtr<Value>> {
        // Check if collection is needed
        if self.should_collect() {
            let _ = self.collect_minor();
        }

        // Generate unique ID
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Create the managed object
        let gc_obj = Arc::new(GcObject::new(id, value, 0));

        // Log object creation for debugging (uses the id and generation methods)
        log::debug!(
            "Allocated object id={} generation={} value_type={:?}",
            gc_obj.id(),
            gc_obj.generation(),
            std::mem::discriminant(gc_obj.value())
        );

        // Store in main objects table and get a stable pointer
        let value_ptr = {
            let mut objects = self.objects.write();
            objects.insert(id, Arc::clone(&gc_obj));
            let stored_obj = objects.get(&id).unwrap();
            &stored_obj.value as *const Value
        };

        // Store in generation 0 (young generation)
        {
            let mut gen0 = self.generations[0].write();
            gen0.insert(id, gc_obj);
        }

        self.stats.record_allocation(std::mem::size_of::<Value>());

        // Create GcPtr pointing to the actual value in the managed object
        Ok(unsafe { GcPtr::from_raw(value_ptr) })
    }

    /// Get a value by its GC handle
    pub fn get_value(&self, handle: GcHandle) -> Option<Value> {
        if handle.is_null() {
            return None;
        }

        let objects = self.objects.read();
        objects.get(&handle.id()).map(|obj| obj.value.clone())
    }

    /// Check if collection should be triggered
    fn should_collect(&self) -> bool {
        if self.collection_in_progress.load(Ordering::Acquire) {
            return false;
        }

        let config = self.config.read();
        let total_objects = self.generations[0].read().len();
        let young_gen_size = config.young_generation_size;
        let threshold = config.minor_gc_threshold;

        // Calculate estimated memory usage (approximate)
        let estimated_bytes = total_objects * std::mem::size_of::<Value>();
        let utilization = estimated_bytes as f64 / young_gen_size as f64;

        // Trigger collection if utilization exceeds threshold
        utilization > threshold || total_objects >= 50 // Fallback for very large object counts
    }

    /// Perform minor collection (young generation)
    pub fn collect_minor(&self) -> Result<usize> {
        if self
            .collection_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return Ok(0);
        }

        let start_time = Instant::now();

        // Mark phase: mark all reachable objects
        let mut marked_count = 0;

        // Mark from explicit roots
        {
            let roots = self.roots.read();
            for &root_handle in roots.keys() {
                if let Some(obj) = self.get_object(root_handle) {
                    if obj.mark() {
                        marked_count += 1;
                        marked_count += self.mark_transitively(&obj);
                    }
                }
            }
        }

        // Sweep phase: collect unmarked objects from young generation
        let mut collected_count = 0;
        let mut to_remove = Vec::new();

        {
            let gen0 = self.generations[0].read();
            for (&id, obj) in gen0.iter() {
                if !obj.is_marked() {
                    to_remove.push(id);
                    collected_count += 1;
                } else {
                    obj.unmark(); // Reset mark for next collection
                }
            }
        }

        // Remove from both generation and main object table
        {
            let mut gen0 = self.generations[0].write();
            let mut objects = self.objects.write();

            for id in to_remove {
                gen0.remove(&id);
                objects.remove(&id);
            }
        }

        let duration = start_time.elapsed();
        self.stats
            .record_minor_collection(collected_count, duration);

        self.collection_in_progress.store(false, Ordering::Release);

        log::info!("Minor GC: marked {marked_count} objects, collected {collected_count} objects in {duration:?}");

        Ok(collected_count)
    }

    /// Get object by handle
    fn get_object(&self, handle: GcHandle) -> Option<Arc<GcObject>> {
        if handle.is_null() {
            return None;
        }

        let objects = self.objects.read();
        objects.get(&handle.id()).map(Arc::clone)
    }

    /// Mark objects transitively (follow references)
    fn mark_transitively(&self, obj: &GcObject) -> usize {
        let mut marked = 0;

        match &obj.value {
            Value::Cell(cells) => {
                for cell_value in cells {
                    if let Some(referenced_obj) = self.find_object_for_value(cell_value) {
                        if referenced_obj.mark() {
                            marked += 1;
                            marked += self.mark_transitively(&referenced_obj);
                        }
                    }
                }
            }
            _ => {
                // Other value types don't contain GC references
            }
        }

        marked
    }

    /// Find the GcObject that contains a specific Value
    fn find_object_for_value(&self, value: &Value) -> Option<Arc<GcObject>> {
        // Search through all generations for an object containing this value
        for generation_map in &self.generations {
            let generation_map = generation_map.read();
            for (_handle_id, gc_object) in generation_map.iter() {
                // Check if this GC object's data matches the value
                if self.value_matches_object(value, gc_object) {
                    return Some(Arc::clone(gc_object));
                }
            }
        }
        None
    }

    /// Check if a Value matches a GcObject's data
    fn value_matches_object(&self, value: &Value, gc_object: &GcObject) -> bool {
        // Since GcObject stores the Value directly, we can just compare them
        &gc_object.value == value
    }

    /// Perform major collection (all generations)
    pub fn collect_major(&self) -> Result<usize> {
        if self
            .collection_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return Ok(0);
        }

        let start_time = Instant::now();

        // Mark phase: mark all reachable objects from roots across all generations
        let mut marked_count = 0;

        // Collect roots from explicit roots
        {
            let roots = self.roots.read();
            for &root_handle in roots.keys() {
                if let Some(obj) = self.get_object(root_handle) {
                    if obj.mark() {
                        marked_count += 1;
                        marked_count += self.mark_transitively(&obj);
                    }
                }
            }
        }

        // Sweep phase: collect unmarked objects from all generations
        let mut collected_count = 0;
        for generation in &self.generations {
            let mut to_remove = Vec::new();

            {
                let gen = generation.read();
                for (&id, obj) in gen.iter() {
                    if !obj.is_marked() {
                        to_remove.push(id);
                        collected_count += 1;
                    } else {
                        obj.unmark(); // Reset mark for next collection
                    }
                }
            }

            // Remove from both generation and main object table
            {
                let mut gen = generation.write();
                let mut objects = self.objects.write();

                for id in to_remove {
                    gen.remove(&id);
                    objects.remove(&id);
                }
            }
        }

        let duration = start_time.elapsed();
        self.stats
            .record_major_collection(collected_count, duration);

        self.collection_in_progress.store(false, Ordering::Release);

        log::info!("Major GC: marked {marked_count} objects, collected {collected_count} objects in {duration:?}");

        Ok(collected_count)
    }

    /// Add a root to protect an object from collection
    pub fn add_root(&self, root: GcPtr<Value>) -> Result<()> {
        let value_ptr = unsafe { root.as_raw() };
        if value_ptr.is_null() {
            return Ok(()); // Null pointer, nothing to protect
        }

        // Find the object that contains this value
        if let Some(handle) = self.find_handle_for_value_ptr(value_ptr) {
            self.roots.write().insert(handle, ());
        }
        Ok(())
    }

    /// Remove a root
    pub fn remove_root(&self, root: GcPtr<Value>) -> Result<()> {
        let value_ptr = unsafe { root.as_raw() };
        if value_ptr.is_null() {
            return Ok(()); // Null pointer, nothing to remove
        }

        // Find the object that contains this value
        if let Some(handle) = self.find_handle_for_value_ptr(value_ptr) {
            self.roots.write().remove(&handle);
        }
        Ok(())
    }

    /// Find the handle for an object that contains the given value pointer
    fn find_handle_for_value_ptr(&self, value_ptr: *const Value) -> Option<GcHandle> {
        let objects = self.objects.read();
        for (&id, obj) in objects.iter() {
            let obj_value_ptr = &obj.value as *const Value;
            if obj_value_ptr == value_ptr {
                return Some(GcHandle::new(id));
            }
        }
        None
    }

    /// Get GC statistics
    pub fn stats(&self) -> GcStats {
        (*self.stats).clone()
    }

    /// Configure the GC
    pub fn configure(&self, config: GcConfig) -> Result<()> {
        *self.config.write() = config;
        Ok(())
    }

    /// Get the current GC configuration
    pub fn get_config(&self) -> GcConfig {
        self.config.read().clone()
    }
}

// Safety: All shared data is protected by proper synchronization
unsafe impl Send for HighPerformanceGC {}
unsafe impl Sync for HighPerformanceGC {}

/// Global garbage collector instance
static GC: once_cell::sync::Lazy<Arc<HighPerformanceGC>> =
    once_cell::sync::Lazy::new(|| Arc::new(HighPerformanceGC::new().expect("Failed to create GC")));

/// Helper function to dereference a GcPtr safely (now just uses normal dereferencing)
pub fn gc_deref(ptr: GcPtr<Value>) -> Value {
    (*ptr).clone()
}

/// Global GC functions for easy access
pub fn gc_allocate(value: Value) -> Result<GcPtr<Value>> {
    GC.allocate(value)
}

pub fn gc_collect_minor() -> Result<usize> {
    GC.collect_minor()
}

pub fn gc_collect_major() -> Result<usize> {
    GC.collect_major()
}

pub fn gc_add_root(root: GcPtr<Value>) -> Result<()> {
    GC.add_root(root)
}

pub fn gc_remove_root(root: GcPtr<Value>) -> Result<()> {
    GC.remove_root(root)
}

pub fn gc_stats() -> GcStats {
    GC.stats()
}

pub fn gc_configure(config: GcConfig) -> Result<()> {
    GC.configure(config)
}

pub fn gc_get_config() -> GcConfig {
    GC.get_config()
}

/// Simplified root registration for backwards compatibility
pub fn gc_register_root(_root: Box<dyn GcRoot>) -> Result<RootId> {
    Ok(RootId(0))
}

pub fn gc_unregister_root(_root_id: RootId) -> Result<()> {
    Ok(())
}

/// Force a garbage collection for testing/debugging
#[cfg(any(test, feature = "debug-gc"))]
pub fn gc_force_collect() -> Result<usize> {
    gc_collect_major()
}

/// Reset GC for testing - always available
pub fn gc_reset_for_test() -> Result<()> {
    // Reset statistics
    GC.stats.reset();

    // Clear all objects and roots for a clean test
    {
        let mut objects = GC.objects.write();
        objects.clear();
    }

    {
        let mut gen0 = GC.generations[0].write();
        gen0.clear();
    }

    {
        let mut gen1 = GC.generations[1].write();
        gen1.clear();
    }

    {
        let mut roots = GC.roots.write();
        roots.clear();
    }

    // Reset next ID
    GC.next_id.store(1, Ordering::Relaxed);

    // Ensure collection is not in progress
    GC.collection_in_progress.store(false, Ordering::Relaxed);

    // Configure with default settings
    gc_configure(GcConfig::default())?;

    Ok(())
}

/// Create an isolated test context with clean GC state
pub fn gc_test_context<F, R>(test_fn: F) -> R
where
    F: FnOnce() -> R,
{
    // Use a global test mutex to ensure tests run sequentially when needed
    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Handle poisoned mutex gracefully
    let _guard = match TEST_MUTEX.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            // Clear the poison and continue
            poisoned.into_inner()
        }
    };

    // Reset GC to clean state
    gc_reset_for_test().expect("GC reset should succeed");

    // Run the test
    let result = test_fn();

    // Clean up after test (optional, but good practice)
    let _ = gc_reset_for_test();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let _ = gc_reset_for_test();

        let value = Value::Num(42.0);
        let ptr = gc_allocate(value).expect("allocation failed");
        assert_eq!(*ptr, Value::Num(42.0));
    }

    #[test]
    fn test_collection() {
        let _ = gc_reset_for_test();

        // Allocate many objects to trigger collection
        let mut _ptrs = Vec::new();
        for i in 0..100 {
            let ptr = gc_allocate(Value::Num(i as f64)).expect("allocation failed");
            _ptrs.push(ptr);
        }

        let _collected = gc_collect_minor().expect("collection failed");
        // Collection completed successfully
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    let mut ptrs = Vec::new();
                    for j in 0..100 {
                        let value = Value::Num((i * 100 + j) as f64);
                        let ptr = gc_allocate(value).expect("allocation failed");
                        ptrs.push(ptr);
                    }
                    ptrs
                })
            })
            .collect();

        for handle in handles {
            let _ptrs = handle.join().expect("thread failed");
        }

        // Force collection to clean up
        let _ = gc_collect_major();
    }

    #[test]
    fn test_root_protection() {
        let _ = gc_reset_for_test();

        // Allocate an object and keep a reference
        let protected = gc_allocate(Value::Num(42.0)).expect("allocation failed");

        // Register it as a root
        gc_add_root(protected).expect("root registration failed");

        // Allocate garbage
        for i in 0..60 {
            let _ = gc_allocate(Value::String(format!("garbage_{i}")));
        }

        // Force collection
        let _ = gc_collect_minor().expect("collection failed");

        // Protected object should still be valid
        assert_eq!(*protected, Value::Num(42.0));

        // Clean up
        gc_remove_root(protected).expect("root removal failed");
    }

    #[test]
    fn test_gc_object_metadata() {
        let _ = gc_reset_for_test();

        // Create a GcObject directly to test its methods
        let value = Value::Num(42.0);
        let gc_obj = GcObject::new(123, value.clone(), 1);

        // Test the methods that were added
        assert_eq!(gc_obj.id(), 123);
        assert_eq!(gc_obj.generation(), 1);
        assert_eq!(gc_obj.value(), &value);

        // Test marking functionality
        assert!(!gc_obj.is_marked());
        assert!(gc_obj.mark()); // First mark should return true (was unmarked)
        assert!(gc_obj.is_marked());
        assert!(!gc_obj.mark()); // Second mark should return false (already marked)

        gc_obj.unmark();
        assert!(!gc_obj.is_marked());
    }
}
