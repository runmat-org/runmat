use parking_lot::{Mutex, RwLock};
// Use a local trait-alias shim to avoid compile-time dependency ordering issues.
// Downstream crates in the workspace provide runmat_builtins; during GC unit tests, we provide a minimal Value.
use runmat_builtins::Value;
use runmat_time::Instant;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

pub mod allocator;
pub mod barriers;
pub mod collector;
pub mod config;
pub use runmat_gc_api::GcHandle;
pub mod generations;
pub mod roots;
pub mod stats;

// Finalizer support
use std::collections::HashMap;

/// A finalizer that runs when a GC-managed Value is collected.
///
/// Finalizers must be fast, non-panicking, and avoid allocating.
pub trait GcFinalizer: Send + Sync {
    fn finalize(&self);
}

static FINALIZERS: once_cell::sync::Lazy<
    parking_lot::Mutex<HashMap<usize, std::sync::Arc<dyn GcFinalizer>>>,
> = once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(HashMap::new()));

/// Register a finalizer for the provided GC handle.
pub fn gc_register_finalizer(ptr: GcHandle, f: std::sync::Arc<dyn GcFinalizer>) -> Result<()> {
    let addr = ptr.addr();
    if addr == 0 {
        return Ok(());
    }
    FINALIZERS.lock().insert(addr, f);
    Ok(())
}

/// Remove any registered finalizer for the provided GC handle.
pub fn gc_unregister_finalizer(ptr: GcHandle) -> Result<()> {
    let addr = ptr.addr();
    if addr == 0 {
        return Ok(());
    }
    FINALIZERS.lock().remove(&addr);
    Ok(())
}

/// Internal: run and remove finalizer for an address if present.
pub(crate) fn gc_run_finalizer_for_addr(addr: usize) {
    if let Some(f) = FINALIZERS.lock().remove(&addr) {
        // Run finalizer; avoid panicking across GC boundaries
        f.finalize();
    }
}

#[cfg(feature = "pointer-compression")]
pub mod compression;

pub use allocator::{AllocatorStats, GenerationalAllocator, SizeClass};
use barriers::WriteBarrierManager;
pub use barriers::{CardTable, WriteBarrier};
pub use collector::MarkSweepCollector;
pub use config::{GcConfig, GcConfigBuilder};
// Re-export unified handle from API crate
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

    #[error("Invalid GC handle: {0}")]
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

// Legacy handle/object table removed in favor of allocator-backed pointers and address-keyed maps

/// Garbage collector with safe handle-based design
pub struct GarbageCollector {
    /// Configuration
    config: Arc<RwLock<GcConfig>>,

    /// Generational allocator (owns heap memory)
    allocator: Mutex<GenerationalAllocator>,

    /// Mark-and-sweep collector
    collector: Mutex<MarkSweepCollector>,

    /// Explicit roots stored as raw addresses (address-keyed)
    root_ptrs: Arc<Mutex<HashSet<usize>>>,

    /// Collection state
    collection_in_progress: AtomicBool,

    /// Active guarded value borrows. Collection is skipped while this is nonzero.
    active_value_borrows: AtomicUsize,

    /// Active mutable guarded value borrow. Mutable access is exclusive.
    active_value_mut_borrow: AtomicBool,

    /// Statistics
    stats: Arc<GcStats>,
}

struct GcValueBorrowGuard<'gc> {
    gc: &'gc GarbageCollector,
    mutable: bool,
}

impl Drop for GcValueBorrowGuard<'_> {
    fn drop(&mut self) {
        self.gc.active_value_borrows.fetch_sub(1, Ordering::AcqRel);
        if self.mutable {
            self.gc
                .active_value_mut_borrow
                .store(false, Ordering::Release);
        }
    }
}

impl GarbageCollector {
    pub fn new() -> Result<Self> {
        Self::with_config(GcConfig::default())
    }

    pub fn with_config(config: GcConfig) -> Result<Self> {
        let allocator = GenerationalAllocator::new(&config);
        let collector = MarkSweepCollector::new(&config);

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            allocator: Mutex::new(allocator),
            collector: Mutex::new(collector),
            root_ptrs: Arc::new(Mutex::new(HashSet::new())),
            collection_in_progress: AtomicBool::new(false),
            active_value_borrows: AtomicUsize::new(0),
            active_value_mut_borrow: AtomicBool::new(false),
            stats: Arc::new(GcStats::new()),
        })
    }

    /// Allocate a new Value object
    pub fn allocate(&self, value: Value) -> Result<GcHandle> {
        // Check if collection is needed
        if self.should_collect() {
            let _ = self.collect_minor();
        }

        // Capture GPU handle if present to attach a finalizer after allocation
        let gpu_handle: Option<runmat_accelerate_api::GpuTensorHandle> =
            if let Value::GpuTensor(h) = &value {
                Some(h.clone())
            } else {
                None
            };

        let mut allocator = self.allocator.lock();
        let ptr = allocator.allocate(value, &self.stats)?;
        let usage = allocator.young_generation_usage();
        let alloc_count = allocator.young_allocations_count();
        let cfg = self.config.read().clone();
        drop(allocator);

        // Register finalizer for GPU tensors so buffers are freed on collection
        if let Some(handle) = gpu_handle {
            struct GpuTensorFinalizer {
                handle: runmat_accelerate_api::GpuTensorHandle,
            }
            impl GcFinalizer for GpuTensorFinalizer {
                fn finalize(&self) {
                    if let Some(p) = runmat_accelerate_api::provider() {
                        let _ = p.free(&self.handle);
                        runmat_accelerate_api::clear_handle_logical(&self.handle);
                    }
                }
            }
            let fin = std::sync::Arc::new(GpuTensorFinalizer { handle });
            let _ = gc_register_finalizer(ptr.clone(), fin);
        }

        // Heuristic triggers:
        // - Utilization threshold
        // - Aggressive mode: periodic minor GC by allocation count to satisfy stress configs
        if usage > cfg.minor_gc_threshold
            || (cfg.minor_gc_threshold <= 0.35 && alloc_count > 0 && alloc_count.is_multiple_of(32))
        {
            let _ = self.collect_minor();
        }
        Ok(ptr)
    }

    fn non_null_value_ptr(ptr: &GcHandle) -> Result<*const Value> {
        Ok(ptr.addr() as *const Value)
    }

    fn validate_value_ptr(&self, raw: *const Value) -> Result<()> {
        let allocator = self.allocator.lock();
        if allocator.find_generation(raw.cast::<u8>()).is_none() {
            return Err(GcError::InvalidPointer(format!(
                "GC value handle {:p} is not owned by the RunMat GC heap",
                raw
            )));
        }
        Ok(())
    }

    fn begin_value_borrow(&self, mutable: bool) -> Result<GcValueBorrowGuard<'_>> {
        if self.collection_in_progress.load(Ordering::Acquire) {
            return Err(GcError::CollectionFailed(
                "cannot borrow GC value during collection".to_string(),
            ));
        }

        if mutable {
            self.active_value_mut_borrow
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| {
                    GcError::SyncError("mutable GC value borrow already active".to_string())
                })?;
            if self.active_value_borrows.load(Ordering::Acquire) != 0 {
                self.active_value_mut_borrow.store(false, Ordering::Release);
                return Err(GcError::SyncError(
                    "cannot mutably borrow GC value while another borrow is active".to_string(),
                ));
            }
        } else if self.active_value_mut_borrow.load(Ordering::Acquire) {
            return Err(GcError::SyncError(
                "cannot immutably borrow GC value during mutable borrow".to_string(),
            ));
        }

        self.active_value_borrows.fetch_add(1, Ordering::AcqRel);
        if self.collection_in_progress.load(Ordering::Acquire) {
            self.active_value_borrows.fetch_sub(1, Ordering::AcqRel);
            if mutable {
                self.active_value_mut_borrow.store(false, Ordering::Release);
            }
            return Err(GcError::CollectionFailed(
                "collection started before GC value borrow could be established".to_string(),
            ));
        }

        Ok(GcValueBorrowGuard { gc: self, mutable })
    }

    /// Access a GC-managed value after validating that the pointer belongs to the GC heap.
    ///
    /// Collection is blocked for the duration of the callback.
    pub fn with_value<R>(&self, ptr: &GcHandle, f: impl FnOnce(&Value) -> R) -> Result<R> {
        let _borrow = self.begin_value_borrow(false)?;
        let raw = Self::non_null_value_ptr(ptr)?;
        self.validate_value_ptr(raw)?;
        Ok(f(unsafe { &*raw }))
    }

    /// Mutably access a GC-managed value after validating that the pointer belongs to the GC heap.
    ///
    /// Collection is blocked for the duration of the callback and mutable access is exclusive.
    pub fn with_value_mut<R>(&self, ptr: &GcHandle, f: impl FnOnce(&mut Value) -> R) -> Result<R> {
        let _borrow = self.begin_value_borrow(true)?;
        let raw = Self::non_null_value_ptr(ptr)? as *mut Value;
        self.validate_value_ptr(raw)?;
        Ok(f(unsafe { &mut *raw }))
    }

    /// Clone a GC-managed value through the guarded access path.
    pub fn clone_value(&self, ptr: &GcHandle) -> Result<Value> {
        self.with_value(ptr, Clone::clone)
    }

    /// Check if collection should be triggered
    fn should_collect(&self) -> bool {
        if self.collection_in_progress.load(Ordering::Acquire) {
            return false;
        }
        if self.active_value_borrows.load(Ordering::Acquire) != 0 {
            return false;
        }

        let threshold = self.config.read().minor_gc_threshold;
        let allocator = self.allocator.lock();
        let utilization = allocator.young_generation_usage();
        utilization > threshold
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
        if self.active_value_borrows.load(Ordering::Acquire) != 0 {
            self.collection_in_progress.store(false, Ordering::Release);
            return Ok(0);
        }

        let start_time = Instant::now();

        // Build combined roots: explicit + external (root scanner + barriers)
        let mut combined_roots: Vec<GcHandle> = Vec::new();
        {
            let roots = self.root_ptrs.lock();
            combined_roots.extend(
                roots
                    .iter()
                    .filter_map(|&addr| NonZeroUsize::new(addr))
                    // SAFETY: explicit roots are recorded from live GC handles
                    // via `GcHandle::addr` in `add_root`.
                    .map(|addr| unsafe { GcHandle::from_addr_unchecked(addr) }),
            );
        }
        // External roots
        if let Ok(mut ext) = ROOT_SCANNER.with(|scanner| scanner.scan_roots()) {
            combined_roots.append(&mut ext);
        }
        combined_roots.extend(gc_barrier_minor_roots());

        // Collect young generation via unified collector/allocator
        let mut allocator = self.allocator.lock();
        let mut collector = self.collector.lock();
        let collected_count =
            collector.collect_young_generation(&mut allocator, &combined_roots, &self.stats)?;

        // Clear dirty cards; keep remembered set for next minor cycle
        WRITE_BARRIERS.clear_after_minor_gc();

        let duration = start_time.elapsed();
        self.stats
            .record_minor_collection(collected_count, duration);

        self.collection_in_progress.store(false, Ordering::Release);

        log::info!("Minor GC: collected {collected_count} objects in {duration:?}");

        Ok(collected_count)
    }

    // /// Get object by handle
    // Address-keyed design: object table lookup removed

    // /// Mark objects transitively (follow references)
    // Transitive marking handled by collector over allocator-backed objects

    // /// Find the GcObject that contains a specific Value
    // Value-equality scans removed in favor of address-keyed marking

    /// Perform major collection (all generations)
    pub fn collect_major(&self) -> Result<usize> {
        if self
            .collection_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return Ok(0);
        }
        if self.active_value_borrows.load(Ordering::Acquire) != 0 {
            self.collection_in_progress.store(false, Ordering::Release);
            return Ok(0);
        }

        let start_time = Instant::now();

        // Build combined roots: explicit + external
        let mut combined_roots: Vec<GcHandle> = Vec::new();
        {
            let roots = self.root_ptrs.lock();
            combined_roots.extend(
                roots
                    .iter()
                    .filter_map(|&addr| NonZeroUsize::new(addr))
                    // SAFETY: explicit roots are recorded from live GC handles
                    // via `GcHandle::addr` in `add_root`.
                    .map(|addr| unsafe { GcHandle::from_addr_unchecked(addr) }),
            );
        }
        if let Ok(mut ext) = ROOT_SCANNER.with(|scanner| scanner.scan_roots()) {
            combined_roots.append(&mut ext);
        }
        combined_roots.extend(gc_barrier_minor_roots());

        let mut allocator = self.allocator.lock();
        let mut collector = self.collector.lock();
        let collected_count =
            collector.collect_all_generations(&mut allocator, &combined_roots, &self.stats)?;

        // Clear all barriers after major GC
        WRITE_BARRIERS.clear_after_major_gc();
        allocator.clear_promotion_state();

        let duration = start_time.elapsed();
        self.stats
            .record_major_collection(collected_count, duration);

        self.collection_in_progress.store(false, Ordering::Release);

        log::info!("Major GC: collected {collected_count} objects in {duration:?}");

        Ok(collected_count)
    }

    /// Add a root to protect an object from collection
    pub fn add_root(&self, root: GcHandle) -> Result<()> {
        let value_ptr = root.addr();
        if value_ptr == 0 {
            return Ok(());
        }
        self.root_ptrs.lock().insert(value_ptr);
        Ok(())
    }

    /// Remove a root
    pub fn remove_root(&self, root: GcHandle) -> Result<()> {
        let value_ptr = root.addr();
        if value_ptr == 0 {
            return Ok(());
        }
        self.root_ptrs.lock().remove(&value_ptr);
        Ok(())
    }

    // /// Find the handle for an object that contains the given value pointer
    // Address-keyed roots: no handle lookup needed
    /// Get GC statistics
    pub fn stats(&self) -> GcStats {
        self.stats.as_ref().clone()
    }

    /// Configure the GC
    pub fn configure(&self, config: GcConfig) -> Result<()> {
        *self.config.write() = config.clone();
        {
            // Rebuild allocator to support changes like num_generations and sizes
            let mut allocator = self.allocator.lock();
            *allocator = GenerationalAllocator::new(&config);
        }
        {
            let mut collector = self.collector.lock();
            collector.reconfigure(&config)?;
        }
        Ok(())
    }

    /// Get the current GC configuration
    pub fn get_config(&self) -> GcConfig {
        self.config.read().clone()
    }
}

// Safety: All shared data is protected by proper synchronization
unsafe impl Send for GarbageCollector {}
unsafe impl Sync for GarbageCollector {}

/// Global garbage collector instance
static GC: once_cell::sync::Lazy<Arc<GarbageCollector>> =
    once_cell::sync::Lazy::new(|| Arc::new(GarbageCollector::new().expect("Failed to create GC")));

thread_local! {
    static ROOT_SCANNER: RootScanner = RootScanner::new();
}

/// Global write barrier manager
static WRITE_BARRIERS: once_cell::sync::Lazy<Arc<WriteBarrierManager>> =
    once_cell::sync::Lazy::new(|| Arc::new(WriteBarrierManager::new(true, false)));

/// Global GC functions for easy access
pub fn gc_allocate(value: Value) -> Result<GcHandle> {
    GC.allocate(value)
}

pub fn gc_collect_minor() -> Result<usize> {
    // Stage external roots inside GC and perform unified minor collection
    GC.collect_minor()
}

pub fn gc_collect_major() -> Result<usize> {
    GC.collect_major()
}

pub fn gc_add_root(root: GcHandle) -> Result<()> {
    GC.add_root(root)
}

pub fn gc_remove_root(root: GcHandle) -> Result<()> {
    GC.remove_root(root)
}

pub fn gc_with_value<R>(ptr: &GcHandle, f: impl FnOnce(&Value) -> R) -> Result<R> {
    GC.with_value(ptr, f)
}

pub fn gc_with_value_mut<R>(ptr: &GcHandle, f: impl FnOnce(&mut Value) -> R) -> Result<R> {
    GC.with_value_mut(ptr, f)
}

pub fn gc_clone_value(ptr: &GcHandle) -> Result<Value> {
    GC.clone_value(ptr)
}

pub fn gc_handle_addr(handle: &GcHandle) -> usize {
    handle.addr()
}

pub fn gc_handle_from_addr(addr: usize) -> Result<GcHandle> {
    let raw = NonZeroUsize::new(addr)
        .ok_or_else(|| GcError::InvalidPointer("null GC value handle address".to_string()))?;
    GC.validate_value_ptr(addr as *const Value)?;
    // SAFETY: the address was validated against the current RunMat GC heap.
    Ok(unsafe { GcHandle::from_addr_unchecked(raw) })
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
pub fn gc_register_root(root: Box<dyn GcRoot>) -> Result<RootId> {
    ROOT_SCANNER.with(|scanner| scanner.register_root(root))
}

pub fn gc_unregister_root(root_id: RootId) -> Result<()> {
    ROOT_SCANNER.with(|scanner| scanner.unregister_root(root_id))
}

/// Record a write for GC barriers (approximate old->young tracking)
pub fn gc_record_write(old: &Value, new: &Value) {
    // Generation-aware barrier: record only if old is logically old and new is young
    let old_ptr = old as *const Value as *const u8;
    let young_ptr = new as *const Value as *const u8;
    // Query allocator logical generations
    let alloc = GC.allocator.lock();
    let old_gen = alloc.logical_generation(old_ptr).unwrap_or(0);
    let new_gen = alloc.logical_generation(young_ptr).unwrap_or(0);
    drop(alloc);
    if old_gen > new_gen {
        WRITE_BARRIERS.record_reference(old_ptr, young_ptr);
    }
}

/// Get barrier-derived roots for minor GC
pub fn gc_barrier_minor_roots() -> Vec<GcHandle> {
    WRITE_BARRIERS
        .get_minor_gc_roots()
        .into_iter()
        .filter_map(|p| NonZeroUsize::new(p as usize))
        // SAFETY: barrier roots are recorded from GC-owned pointer addresses by
        // the write-barrier manager.
        .map(|addr| unsafe { GcHandle::from_addr_unchecked(addr) })
        .collect()
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

    // Reset allocator/collector and roots
    {
        let config = GcConfig::default();
        *GC.config.write() = config.clone();
        let mut alloc = GC.allocator.lock();
        *alloc = GenerationalAllocator::new(&config);
        let mut coll = GC.collector.lock();
        *coll = MarkSweepCollector::new(&config);
    }

    {
        let mut roots = GC.root_ptrs.lock();
        roots.clear();
    }

    // Ensure collection is not in progress
    GC.collection_in_progress.store(false, Ordering::Relaxed);
    GC.active_value_borrows.store(0, Ordering::Relaxed);
    GC.active_value_mut_borrow.store(false, Ordering::Relaxed);

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
        gc_test_context(|| {
            let value = Value::Num(42.0);
            let ptr = gc_allocate(value).expect("allocation failed");
            assert_eq!(
                gc_clone_value(&ptr).expect("valid GC handle"),
                Value::Num(42.0)
            );
        });
    }

    #[test]
    fn test_collection() {
        gc_test_context(|| {
            let mut _ptrs = Vec::new();
            for i in 0..100 {
                let ptr = gc_allocate(Value::Num(i as f64)).expect("allocation failed");
                _ptrs.push(ptr);
            }

            let _collected = gc_collect_minor().expect("collection failed");
            drop(_ptrs);
        });
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        gc_test_context(|| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    thread::spawn(move || {
                        let mut ptrs = Vec::new();
                        for j in 0..100 {
                            let value = Value::Num((i * 100 + j) as f64);
                            let ptr = gc_allocate(value).expect("allocation failed");
                            ptrs.push(ptr);
                        }
                        ptrs.len()
                    })
                })
                .collect();

            for handle in handles {
                assert_eq!(handle.join().expect("thread failed"), 100);
            }

            let _ = gc_collect_major();
        });
    }

    #[test]
    fn test_root_protection() {
        gc_test_context(|| {
            let protected = gc_allocate(Value::Num(42.0)).expect("allocation failed");
            gc_add_root(protected.clone()).expect("root registration failed");

            for i in 0..60 {
                let _ = gc_allocate(Value::String(format!("garbage_{i}")));
            }

            let _ = gc_collect_minor().expect("collection failed");
            assert_eq!(
                gc_clone_value(&protected).expect("valid GC handle"),
                Value::Num(42.0)
            );

            gc_remove_root(protected).expect("root removal failed");
        });
    }

    #[test]
    fn test_gc_allocation_and_roots() {
        gc_test_context(|| {
            let v = gc_allocate(Value::Num(7.0)).expect("allocation failed");
            assert_eq!(
                gc_clone_value(&v).expect("valid GC handle"),
                Value::Num(7.0)
            );

            gc_add_root(v.clone()).expect("root add failed");
            let _ = gc_collect_minor().expect("collection failed");
            assert_eq!(
                gc_clone_value(&v).expect("valid GC handle"),
                Value::Num(7.0)
            );

            gc_remove_root(v).expect("root remove failed");
        });
    }

    #[test]
    fn guarded_access_rejects_non_gc_pointer() {
        gc_test_context(|| {
            let raw = Box::into_raw(Box::new(Value::Num(1.0)));
            let ptr = unsafe {
                GcHandle::from_addr_unchecked(
                    NonZeroUsize::new(raw as usize).expect("non-null test pointer"),
                )
            };

            let err = gc_clone_value(&ptr).expect_err("non-GC handle should be rejected");
            assert!(matches!(err, GcError::InvalidPointer(_)));

            unsafe {
                drop(Box::from_raw(raw));
            }
        });
    }

    #[test]
    fn guarded_access_blocks_nested_collection() {
        gc_test_context(|| {
            let ptr = gc_allocate(Value::Num(1.0)).expect("allocation failed");

            let collected = gc_with_value(&ptr, |_| {
                gc_collect_minor().expect("nested collection should be skipped")
            })
            .expect("guarded access should succeed");

            assert_eq!(collected, 0);
            assert_eq!(
                gc_clone_value(&ptr).expect("value should remain live"),
                Value::Num(1.0)
            );
        });
    }

    #[test]
    fn guarded_mut_access_is_exclusive() {
        gc_test_context(|| {
            let ptr = gc_allocate(Value::Num(1.0)).expect("allocation failed");

            let nested = gc_with_value_mut(&ptr, |_| gc_with_value_mut(&ptr, |_| ()))
                .expect("outer mutable borrow should succeed");

            assert!(matches!(nested, Err(GcError::SyncError(_))));
        });
    }
}
