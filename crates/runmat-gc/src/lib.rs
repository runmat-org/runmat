#![forbid(unsafe_op_in_unsafe_fn)]

use allocator::GcAllocationKind;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use runmat_time::Instant;
use runmat_value::Value;
use std::cell::Cell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::ThreadId;
use thiserror::Error;

pub mod allocator;
pub mod barriers;
pub mod collector;
pub mod config;
pub use runmat_gc_api::{GcHandle, GcRoot, RootId, RootInfo, RootScannerStats, Trace, Tracer};
pub mod generations;
mod lexical_activation;
pub use lexical_activation::{
    LexicalActivation, LexicalBindingMutationError, LexicalReservationOutcome, LexicalSlotState,
    LexicalValueOwner, PreparedLexicalTable, RetiredLexicalTable,
};
mod reachability;
pub mod roots;
pub use reachability::{gc_with_registered_root_reachability, RegisteredRootReachability};
mod root_admission;
pub use root_admission::{
    PreparedRootLease, PreparedRootPlan, PreparedRootQuote, RootAdmissionError, RootTraceFailure,
};
mod root_retirement;
pub use root_retirement::{RootRetirement, RootRetirementObserver};
mod admitted_allocation;
mod retirement;
pub use admitted_allocation::{
    gc_allocate_prepared_rooted, GcRootedAllocationError, PreparedGcAllocation,
};
pub mod stats;
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod test_support;

// Finalizer support
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct ExternalRootProvider {
    roots: fn() -> Vec<GcHandle>,
    roots_on_other_threads: fn() -> bool,
}

static EXTERNAL_ROOT_PROVIDERS: once_cell::sync::Lazy<
    RwLock<HashMap<&'static str, ExternalRootProvider>>,
> = once_cell::sync::Lazy::new(|| RwLock::new(HashMap::new()));

/// Root publication and collection admission cannot cross a last-owner
/// inspection. The inspection callback may only update a pre-rooted lifecycle
/// state; it must not register roots or execute user code under this guard.
static ROOT_ADMISSION: RwLock<()> = RwLock::new(());

/// Register a runtime-owned root source without making the collector depend on
/// that runtime domain. Re-registering the same stable name replaces it.
pub fn register_external_root_provider(
    name: &'static str,
    roots: fn() -> Vec<GcHandle>,
    roots_on_other_threads: fn() -> bool,
) {
    let _admission = ROOT_ADMISSION.write();
    EXTERNAL_ROOT_PROVIDERS.write().insert(
        name,
        ExternalRootProvider {
            roots,
            roots_on_other_threads,
        },
    );
}

fn external_roots() -> Vec<GcHandle> {
    let providers = EXTERNAL_ROOT_PROVIDERS
        .read()
        .values()
        .copied()
        .collect::<Vec<_>>();
    providers
        .into_iter()
        .flat_map(|provider| (provider.roots)())
        .collect()
}

fn external_roots_exist_on_other_threads() -> bool {
    let providers = EXTERNAL_ROOT_PROVIDERS
        .read()
        .values()
        .copied()
        .collect::<Vec<_>>();
    providers
        .into_iter()
        .any(|provider| (provider.roots_on_other_threads)())
}

/// A finalizer that runs when a GC-managed Value is collected.
///
/// Finalizers must be fast, non-panicking, and avoid allocating.
pub trait GcFinalizer: Send + Sync {
    fn finalize(&self);
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GcIdentity {
    addr: usize,
    epoch: usize,
}

impl From<GcHandle> for GcIdentity {
    fn from(handle: GcHandle) -> Self {
        Self {
            addr: handle.addr(),
            epoch: handle.epoch(),
        }
    }
}

static FINALIZERS: once_cell::sync::Lazy<
    parking_lot::Mutex<HashMap<GcIdentity, std::sync::Arc<dyn GcFinalizer>>>,
> = once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(HashMap::new()));

/// Register a finalizer for the provided GC handle.
pub fn gc_register_finalizer(ptr: GcHandle, f: std::sync::Arc<dyn GcFinalizer>) -> Result<()> {
    let previous = FINALIZERS.lock().insert(ptr.into(), f);
    drop(previous);
    Ok(())
}

/// Remove any registered finalizer for the provided GC handle.
pub fn gc_unregister_finalizer(ptr: GcHandle) -> Result<()> {
    let removed = FINALIZERS.lock().remove(&ptr.into());
    drop(removed);
    Ok(())
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
pub use roots::{GlobalRoot, RootScanner, StackRoot, ValueRoot, VariableArrayRoot};
pub use stats::{CollectionEvent, CollectionType, GcStats};

#[cfg(feature = "pointer-compression")]
pub use compression::*;

/// Errors that can occur during garbage collection
#[derive(Error, Debug, PartialEq, Eq)]
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

    /// Explicit roots stored as identity handles with registration counts.
    root_ptrs: Mutex<root_admission::RootRegistry>,

    /// Collection state
    collection_in_progress: AtomicBool,

    /// Monotonic proof that a successful collection has inspected the heap.
    /// It never wraps: collection fails closed before exhaustion.
    collection_epoch: AtomicU64,

    /// Active allocations whose just-created handles have not been returned or rooted.
    allocation_in_progress: AtomicUsize,

    /// Active guarded value borrows. Collection is skipped while this is nonzero.
    active_value_borrows: AtomicUsize,

    /// Active mutable guarded value borrow. Mutable access is exclusive.
    active_value_mut_borrow: AtomicBool,

    /// Runtime borrow gate for guarded `Value` references.
    value_access: RwLock<()>,

    /// Statistics
    stats: Arc<GcStats>,
}

enum GcValueAccessGuard<'gc> {
    Shared { _guard: RwLockReadGuard<'gc, ()> },
    Exclusive { _guard: RwLockWriteGuard<'gc, ()> },
}

#[derive(Clone, Copy, Default)]
struct ThreadBorrowState {
    shared: usize,
    mutable: bool,
}

thread_local! {
    static THREAD_BORROW_STATE: Cell<ThreadBorrowState> =
        const { Cell::new(ThreadBorrowState { shared: 0, mutable: false }) };
}

struct GcValueBorrowGuard<'gc> {
    gc: &'gc GarbageCollector,
    mutable: bool,
    _access: GcValueAccessGuard<'gc>,
}

impl Drop for GcValueBorrowGuard<'_> {
    fn drop(&mut self) {
        THREAD_BORROW_STATE.with(|state| {
            let mut current = state.get();
            if self.mutable {
                current.mutable = false;
            } else {
                current.shared = current.shared.saturating_sub(1);
            }
            state.set(current);
        });
        self.gc.active_value_borrows.fetch_sub(1, Ordering::AcqRel);
        if self.mutable {
            self.gc
                .active_value_mut_borrow
                .store(false, Ordering::Release);
        }
    }
}

/// Immutable guarded access to a GC-managed `Value`.
///
/// The guard keeps collection from reclaiming the value while the borrow is
/// active. Unlike `GcHandle`, this type may dereference because its lifetime is
/// tied to an active GC borrow guard.
pub struct GcValueRef<'gc> {
    raw: *const Value,
    _borrow: GcValueBorrowGuard<'gc>,
}

impl Deref for GcValueRef<'_> {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `raw` was validated as a GC-owned Value before this guard was
        // constructed, and `_borrow` prevents collection for the guard lifetime.
        unsafe { &*self.raw }
    }
}

/// Mutable guarded access to a GC-managed `Value`.
///
/// The guard keeps collection from reclaiming the value and enforces exclusive
/// mutable access through the GC borrow gate.
pub struct GcValueMut<'gc> {
    owner: GcHandle,
    raw: *mut Value,
    _borrow: GcValueBorrowGuard<'gc>,
}

impl Drop for GcValueMut<'_> {
    fn drop(&mut self) {
        // Record old->young edges for the final value stored through this
        // mutable guard before the borrow gate is released.
        gc_record_handle_write(&self.owner, unsafe { &*self.raw });
    }
}

impl Deref for GcValueMut<'_> {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `raw` was validated as a GC-owned Value before this guard was
        // constructed, and `_borrow` prevents collection for the guard lifetime.
        unsafe { &*self.raw }
    }
}

impl DerefMut for GcValueMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: `raw` was validated as a GC-owned Value before this guard was
        // constructed, and `_borrow` enforces exclusive mutable access.
        unsafe { &mut *self.raw }
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
            root_ptrs: Mutex::new(root_admission::RootRegistry::new()),
            collection_in_progress: AtomicBool::new(false),
            collection_epoch: AtomicU64::new(0),
            allocation_in_progress: AtomicUsize::new(0),
            active_value_borrows: AtomicUsize::new(0),
            active_value_mut_borrow: AtomicBool::new(false),
            value_access: RwLock::new(()),
            stats: Arc::new(GcStats::new()),
        })
    }

    fn allocate_inner(&self, value: Value) -> Result<(GcHandle, bool)> {
        let mut allocator = self.allocator.lock();
        let ptr = match allocator.allocate_preserving_value(value, &self.stats) {
            Ok(ptr) => ptr,
            Err(failure) => {
                drop(allocator);
                let allocator::AllocationFailure { error, value } = *failure;
                drop(value);
                return Err(error);
            }
        };
        let should_collect = self.allocation_should_collect(&allocator);
        drop(allocator);
        Ok((ptr, should_collect))
    }

    /// The same post-allocation policy applies to Values and typed activations.
    /// Caller retains the allocator lock only while observing its counters.
    fn allocation_should_collect(&self, allocator: &allocator::GenerationalAllocator) -> bool {
        let usage = allocator.young_generation_usage();
        let alloc_count = allocator.young_allocations_count();
        let cfg = self.config.read();

        // Heuristic triggers:
        // - Utilization threshold
        // - Aggressive mode: periodic minor GC by allocation count to satisfy stress configs
        usage > cfg.minor_gc_threshold
            || (cfg.minor_gc_threshold <= 0.35 && alloc_count > 0 && alloc_count.is_multiple_of(32))
    }

    /// Allocate a new Value object
    pub fn allocate(&self, value: Value) -> Result<GcHandle> {
        // Check if collection is needed
        if self.should_collect() {
            let _ = self.collect_minor();
        }

        let gate = AllocationGate::enter(self);
        let (ptr, should_collect) = self.allocate_inner(value)?;
        if should_collect {
            self.add_root(ptr)?;
            drop(gate);
            let collect_result = self.collect_minor();
            let remove_result = self.remove_root(ptr);
            collect_result?;
            remove_result?;
        }
        Ok(ptr)
    }

    fn non_null_value_ptr(ptr: &GcHandle) -> Result<*const Value> {
        // SAFETY: this exposes the handle's preserved pointer token. Callers
        // validate heap ownership before dereferencing it.
        Ok(unsafe { ptr.as_ptr_unchecked().cast::<Value>().as_ptr() })
    }

    fn validate_handle_kind(&self, handle: &GcHandle, kind: GcAllocationKind) -> Result<()> {
        let allocator = self.allocator.lock();
        if allocator.allocation_kind(handle) != Some(kind) {
            return Err(GcError::InvalidPointer(format!(
                "GC handle {:p} is not a live {:?} in the RunMat GC heap",
                handle.addr() as *const u8,
                kind,
            )));
        }
        Ok(())
    }

    fn validate_value_handle(&self, handle: &GcHandle, _raw: *const Value) -> Result<()> {
        self.validate_handle_kind(handle, GcAllocationKind::Value)
    }

    fn validate_live_handle(&self, handle: &GcHandle) -> Result<()> {
        if self.allocator.lock().is_live_handle(handle) {
            Ok(())
        } else {
            Err(GcError::InvalidPointer(format!(
                "GC handle {:p} is not live in the RunMat GC heap",
                handle.addr() as *const u8,
            )))
        }
    }

    fn begin_value_borrow(&self, mutable: bool) -> Result<GcValueBorrowGuard<'_>> {
        if self.collection_in_progress.load(Ordering::Acquire) {
            return Err(GcError::CollectionFailed(
                "cannot borrow GC value during collection".to_string(),
            ));
        }

        THREAD_BORROW_STATE.with(|state| {
            let current = state.get();
            if mutable && (current.mutable || current.shared != 0) {
                return Err(GcError::SyncError(
                    "cannot mutably borrow GC value while another borrow is active".to_string(),
                ));
            }
            if !mutable && current.mutable {
                return Err(GcError::SyncError(
                    "cannot immutably borrow GC value during mutable borrow".to_string(),
                ));
            }
            Ok(())
        })?;

        let access = if mutable {
            let guard = self.value_access.write();
            self.active_value_mut_borrow.store(true, Ordering::Release);
            GcValueAccessGuard::Exclusive { _guard: guard }
        } else {
            let guard = self.value_access.read();
            GcValueAccessGuard::Shared { _guard: guard }
        };

        THREAD_BORROW_STATE.with(|state| {
            let mut current = state.get();
            if mutable {
                current.mutable = true;
            } else {
                current.shared += 1;
            }
            state.set(current);
        });
        self.active_value_borrows.fetch_add(1, Ordering::AcqRel);
        if self.collection_in_progress.load(Ordering::Acquire) {
            THREAD_BORROW_STATE.with(|state| {
                let mut current = state.get();
                if mutable {
                    current.mutable = false;
                } else {
                    current.shared = current.shared.saturating_sub(1);
                }
                state.set(current);
            });
            self.active_value_borrows.fetch_sub(1, Ordering::AcqRel);
            if mutable {
                self.active_value_mut_borrow.store(false, Ordering::Release);
            }
            return Err(GcError::CollectionFailed(
                "collection started before GC value borrow could be established".to_string(),
            ));
        }

        Ok(GcValueBorrowGuard {
            gc: self,
            mutable,
            _access: access,
        })
    }

    /// Access a GC-managed value after validating that the pointer belongs to the GC heap.
    ///
    /// Collection is blocked for the duration of the callback.
    pub fn with_value<R>(&self, ptr: &GcHandle, f: impl FnOnce(&Value) -> R) -> Result<R> {
        let value = self.read_value(ptr)?;
        Ok(f(&value))
    }

    /// Mutably access a GC-managed value after validating that the pointer belongs to the GC heap.
    ///
    /// Collection is blocked for the duration of the callback and mutable access is exclusive.
    pub fn with_value_mut<R>(&self, ptr: &GcHandle, f: impl FnOnce(&mut Value) -> R) -> Result<R> {
        let mut value = self.write_value(ptr)?;
        Ok(f(&mut value))
    }

    fn with_lexical_activation<R>(
        &self,
        handle: &GcHandle,
        f: impl FnOnce(&LexicalActivation) -> R,
    ) -> Result<R> {
        let _borrow = self.begin_value_borrow(false)?;
        self.validate_handle_kind(handle, GcAllocationKind::LexicalActivation)?;
        // SAFETY: the live kind check and GC borrow guard establish the typed
        // payload and prevent collection for the whole callback.
        Ok(f(unsafe { &*(handle.addr() as *const LexicalActivation) }))
    }

    fn with_lexical_activation_mut<R>(
        &self,
        handle: &GcHandle,
        f: impl FnOnce(&mut LexicalActivation) -> R,
    ) -> Result<R> {
        let _borrow = self.begin_value_borrow(true)?;
        self.validate_handle_kind(handle, GcAllocationKind::LexicalActivation)?;
        // SAFETY: the live kind check and exclusive GC borrow guard establish
        // the typed payload and prevent concurrent mutation or collection.
        Ok(f(unsafe {
            &mut *(handle.addr() as *mut LexicalActivation)
        }))
    }

    /// Borrow a GC-managed value through an explicit guard.
    pub fn read_value(&self, ptr: &GcHandle) -> Result<GcValueRef<'_>> {
        let borrow = self.begin_value_borrow(false)?;
        let raw = Self::non_null_value_ptr(ptr)?;
        self.validate_value_handle(ptr, raw)?;
        Ok(GcValueRef {
            raw,
            _borrow: borrow,
        })
    }

    /// Mutably borrow a GC-managed value through an explicit guard.
    pub fn write_value(&self, ptr: &GcHandle) -> Result<GcValueMut<'_>> {
        let borrow = self.begin_value_borrow(true)?;
        let raw = Self::non_null_value_ptr(ptr)? as *mut Value;
        self.validate_value_handle(ptr, raw)?;
        Ok(GcValueMut {
            owner: *ptr,
            raw,
            _borrow: borrow,
        })
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
        let collection_guard = CollectionInProgressGuard::new(&self.collection_in_progress);
        if self.collection_epoch.load(Ordering::Acquire) == u64::MAX {
            return Err(GcError::CollectionFailed(
                "GC collection generation exhausted".into(),
            ));
        }
        let admission = ROOT_ADMISSION.write();
        if self.allocation_in_progress.load(Ordering::Acquire) != 0 {
            return Ok(0);
        }
        if self.active_value_borrows.load(Ordering::Acquire) != 0 {
            return Ok(0);
        }
        // Registered VM stack/vars roots live in a thread-local scanner. Until
        // RunMat has stop-the-world safepoints or synchronized root snapshots,
        // cross-thread collection must defer rather than miss another thread's
        // interpreter roots.
        if registered_roots_exist_on_other_threads() || external_roots_exist_on_other_threads() {
            return Ok(0);
        }

        let start_time = Instant::now();

        // Build combined roots: explicit + external (root scanner + barriers)
        let mut combined_roots: Vec<GcHandle> = Vec::new();
        {
            let roots = self.root_ptrs.lock();
            combined_roots.extend(roots.keys().cloned());
        }
        // External roots
        prune_inactive_thread_roots();
        if let Ok(mut ext) = ROOT_SCANNER.with(|scanner| scanner.scan_roots()) {
            combined_roots.append(&mut ext);
        }
        combined_roots.extend(external_roots());
        combined_roots.extend(gc_barrier_minor_roots());

        // Collect young generation via unified collector/allocator
        let mut allocator = self.allocator.lock();
        let mut collector = self.collector.lock();
        let retired =
            collector.collect_young_generation(&mut allocator, &combined_roots, &self.stats)?;
        let collected_count = retired.len();
        drop(collector);
        drop(allocator);

        // Clear dirty cards; keep remembered set for next minor cycle
        WRITE_BARRIERS.clear_after_minor_gc();
        self.collection_epoch.fetch_add(1, Ordering::AcqRel);
        // Retiring an owning payload may invoke a provider callback that
        // allocates. Heap bookkeeping is complete; do not hold its entry gate.
        drop(collection_guard);
        drop(admission);
        drop(retired);

        let duration = start_time.elapsed();
        self.stats
            .record_minor_collection(collected_count, duration);

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
        let collection_guard = CollectionInProgressGuard::new(&self.collection_in_progress);
        if self.collection_epoch.load(Ordering::Acquire) == u64::MAX {
            return Err(GcError::CollectionFailed(
                "GC collection generation exhausted".into(),
            ));
        }
        let admission = ROOT_ADMISSION.write();
        if self.allocation_in_progress.load(Ordering::Acquire) != 0 {
            return Ok(0);
        }
        if self.active_value_borrows.load(Ordering::Acquire) != 0 {
            return Ok(0);
        }
        // Same thread-local root visibility rule as minor collection.
        if registered_roots_exist_on_other_threads() || external_roots_exist_on_other_threads() {
            return Ok(0);
        }

        let start_time = Instant::now();

        // Build combined roots: explicit + external
        let mut combined_roots: Vec<GcHandle> = Vec::new();
        {
            let roots = self.root_ptrs.lock();
            combined_roots.extend(roots.keys().cloned());
        }
        prune_inactive_thread_roots();
        if let Ok(mut ext) = ROOT_SCANNER.with(|scanner| scanner.scan_roots()) {
            combined_roots.append(&mut ext);
        }
        combined_roots.extend(external_roots());
        combined_roots.extend(gc_barrier_minor_roots());

        let mut allocator = self.allocator.lock();
        let mut collector = self.collector.lock();
        let retired =
            collector.collect_all_generations(&mut allocator, &combined_roots, &self.stats)?;
        let collected_count = retired.len();

        // Clear all barriers after major GC
        WRITE_BARRIERS.clear_after_major_gc();
        allocator.clear_promotion_state();
        drop(collector);
        drop(allocator);
        self.collection_epoch.fetch_add(1, Ordering::AcqRel);
        drop(collection_guard);
        drop(admission);
        drop(retired);

        let duration = start_time.elapsed();
        self.stats
            .record_major_collection(collected_count, duration);

        log::info!("Major GC: collected {collected_count} objects in {duration:?}");

        Ok(collected_count)
    }

    /// Add a root to protect an object from collection
    pub fn add_root(&self, root: GcHandle) -> Result<()> {
        let admission = ROOT_ADMISSION.write();
        self.validate_live_handle(&root)?;
        let mut roots = self.root_ptrs.lock();
        let retired = roots.add_unbounded(root)?;
        drop(roots);
        drop(admission);
        drop(retired);
        Ok(())
    }

    /// Remove a root
    pub fn remove_root(&self, root: GcHandle) -> Result<()> {
        {
            let _admission = ROOT_ADMISSION.write();
            let mut roots = self.root_ptrs.lock();
            roots.remove(root);
        }
        root_retirement::notify(RootRetirement::Explicit(root));
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
        let mut allocator = self.allocator.lock();
        let mut collector = self.collector.lock();
        if config.num_generations != allocator.generation_count() {
            if allocator.has_live_allocations() {
                return Err(GcError::ConfigError(
                    "Cannot change the number of generations while GC allocations are live"
                        .to_string(),
                ));
            }
            *allocator = GenerationalAllocator::new(&config);
            *collector = MarkSweepCollector::new(&config);
        } else {
            allocator.reconfigure(&config)?;
            collector.reconfigure(&config)?;
        }
        drop(collector);
        drop(allocator);
        *self.config.write() = config;
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
    static ACTIVE_ROOT_THREAD_GUARD: ActiveRootThreadGuard = const { ActiveRootThreadGuard };
}

static ACTIVE_ROOT_THREADS: once_cell::sync::Lazy<Mutex<HashMap<ThreadId, usize>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Global write barrier manager
static WRITE_BARRIERS: once_cell::sync::Lazy<Arc<WriteBarrierManager>> =
    once_cell::sync::Lazy::new(|| Arc::new(WriteBarrierManager::new(true, false)));

struct ActiveRootThreadGuard;

impl Drop for ActiveRootThreadGuard {
    fn drop(&mut self) {
        ACTIVE_ROOT_THREADS
            .lock()
            .remove(&std::thread::current().id());
    }
}

fn register_active_root_thread() {
    ACTIVE_ROOT_THREAD_GUARD.with(|_| {});
    let thread_id = std::thread::current().id();
    let mut roots = ACTIVE_ROOT_THREADS.lock();
    *roots.entry(thread_id).or_insert(0) += 1;
}

fn unregister_active_root_thread() {
    let thread_id = std::thread::current().id();
    let mut roots = ACTIVE_ROOT_THREADS.lock();
    if let Some(count) = roots.get_mut(&thread_id) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            roots.remove(&thread_id);
        }
    }
}

fn registered_roots_exist_on_other_threads() -> bool {
    let current = std::thread::current().id();
    ACTIVE_ROOT_THREADS
        .lock()
        .iter()
        .any(|(thread_id, count)| *thread_id != current && *count > 0)
}

fn prune_inactive_thread_roots() {
    let removed = ROOT_SCANNER.with(|scanner| scanner.remove_inactive_roots());
    for _ in 0..removed {
        unregister_active_root_thread();
    }
}

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

/// RAII guard for an explicitly registered GC root.
///
/// Prefer this over manually pairing `gc_add_root` and `gc_remove_root` when a
/// handle must stay alive for a lexical scope.
#[derive(Debug)]
pub struct ExplicitRoot {
    handle: Option<GcHandle>,
}

impl ExplicitRoot {
    pub fn new(handle: GcHandle) -> Result<Self> {
        gc_add_root(handle)?;
        Ok(Self {
            handle: Some(handle),
        })
    }

    pub fn handle(&self) -> GcHandle {
        self.handle.expect("explicit root already removed")
    }

    pub fn unroot(mut self) -> Result<GcHandle> {
        let handle = self.handle.take().expect("explicit root already removed");
        gc_remove_root(handle)?;
        Ok(handle)
    }
}

impl Drop for ExplicitRoot {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = gc_remove_root(handle);
        }
    }
}

pub fn gc_root(handle: GcHandle) -> Result<ExplicitRoot> {
    ExplicitRoot::new(handle)
}

struct CollectionInProgressGuard<'a> {
    active: &'a AtomicBool,
}

impl<'a> CollectionInProgressGuard<'a> {
    fn new(active: &'a AtomicBool) -> Self {
        Self { active }
    }
}

impl Drop for CollectionInProgressGuard<'_> {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
    }
}

struct AllocationGate<'a> {
    gc: &'a GarbageCollector,
}

impl<'a> AllocationGate<'a> {
    fn enter(gc: &'a GarbageCollector) -> Self {
        loop {
            while gc.collection_in_progress.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            gc.allocation_in_progress.fetch_add(1, Ordering::AcqRel);
            if !gc.collection_in_progress.load(Ordering::Acquire) {
                return Self { gc };
            }
            gc.allocation_in_progress.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for AllocationGate<'_> {
    fn drop(&mut self) {
        self.gc
            .allocation_in_progress
            .fetch_sub(1, Ordering::AcqRel);
    }
}

/// Allocate a GC value and register it as an explicit root before collection can run.
pub fn gc_allocate_rooted(value: Value) -> Result<ExplicitRoot> {
    // Unlike prepared allocation, incoming children need not have independent
    // roots. Do not collect until their new parent has its actual root.
    let gate = AllocationGate::enter(&GC);
    let (handle, should_collect) = GC.allocate_inner(value)?;
    GC.add_root(handle)?;
    let root = ExplicitRoot {
        handle: Some(handle),
    };
    drop(gate);
    if should_collect {
        GC.collect_minor()?;
    }
    Ok(root)
}

/// Allocate a private lexical activation and root it before collection may
/// observe the new handle. No language Value is constructed at its address.
pub fn gc_allocate_lexical_activation_rooted(
    activation: LexicalActivation,
) -> Result<ExplicitRoot> {
    let gate = AllocationGate::enter(&GC);
    let mut allocator = GC.allocator.lock();
    let handle = match allocator.allocate_lexical_activation(activation, &GC.stats) {
        Ok(handle) => handle,
        Err((error, activation)) => {
            drop(allocator);
            drop(activation);
            return Err(error);
        }
    };
    let should_collect = GC.allocation_should_collect(&allocator);
    drop(allocator);
    GC.add_root(handle)?;
    let root = ExplicitRoot {
        handle: Some(handle),
    };
    drop(gate);
    if should_collect {
        GC.collect_minor()?;
    }
    Ok(root)
}

pub fn gc_with_lexical_activation<R>(
    handle: &GcHandle,
    f: impl FnOnce(&LexicalActivation) -> R,
) -> Result<R> {
    GC.with_lexical_activation(handle, f)
}

pub fn gc_with_lexical_activation_mut<R>(
    handle: &GcHandle,
    f: impl FnOnce(&mut LexicalActivation) -> R,
) -> Result<R> {
    GC.with_lexical_activation_mut(handle, f)
}

/// A short, prevalidated exclusive phase for a prepared lexical batch. It is
/// acquired before any paired external workspace commit, so no collection or
/// competing GC mutation can invalidate the prepared activation handles in
/// the allocation-free commit closure.
pub struct LexicalMutationGuard<'a> {
    handles: &'a [GcHandle],
    _borrow: GcValueBorrowGuard<'static>,
}

impl LexicalMutationGuard<'_> {
    /// Visit a handle from the exact slice validated when this guard was
    /// acquired. The index is private batch metadata, not a new handle
    /// lookup that could fail after a paired workspace commit.
    pub fn with_prevalidated_activation<R>(
        &mut self,
        index: usize,
        visit: impl FnOnce(&mut LexicalActivation) -> R,
    ) -> R {
        let handle = self.handles[index];
        // SAFETY: validation established the live typed allocation, this
        // guard blocks collection, and &mut self serializes all references.
        visit(unsafe { &mut *(handle.addr() as *mut LexicalActivation) })
    }
}

pub fn gc_lock_lexical_mutations(handles: &[GcHandle]) -> Result<LexicalMutationGuard<'_>> {
    let borrow = GC.begin_value_borrow(true)?;
    {
        let allocator = GC.allocator.lock();
        for handle in handles {
            if allocator.allocation_kind(handle) != Some(GcAllocationKind::LexicalActivation) {
                return Err(GcError::InvalidPointer(format!(
                    "GC lexical activation handle {:p} is not live",
                    handle.addr() as *const u8,
                )));
            }
        }
    }
    Ok(LexicalMutationGuard {
        handles,
        _borrow: borrow,
    })
}

pub fn gc_with_value<R>(ptr: &GcHandle, f: impl FnOnce(&Value) -> R) -> Result<R> {
    GC.with_value(ptr, f)
}

pub fn gc_with_value_mut<R>(ptr: &GcHandle, f: impl FnOnce(&mut Value) -> R) -> Result<R> {
    GC.with_value_mut(ptr, f)
}

pub fn gc_read_value(ptr: &GcHandle) -> Result<GcValueRef<'static>> {
    GC.read_value(ptr)
}

pub fn gc_write_value(ptr: &GcHandle) -> Result<GcValueMut<'static>> {
    GC.write_value(ptr)
}

pub fn gc_clone_value(ptr: &GcHandle) -> Result<Value> {
    GC.clone_value(ptr)
}

pub fn gc_handle_addr(handle: &GcHandle) -> usize {
    handle.addr()
}

pub fn gc_stats() -> GcStats {
    GC.stats()
}

/// Successful heap collections only; skipped/reentrant collection attempts
/// leave this generation unchanged. Consumers may recheck retired identities
/// after it advances, but it is not evidence that a specific handle died.
pub fn gc_collection_epoch() -> u64 {
    GC.collection_epoch.load(Ordering::Acquire)
}

pub fn gc_configure(config: GcConfig) -> Result<()> {
    GC.configure(config)
}

pub fn gc_get_config() -> GcConfig {
    GC.get_config()
}

/// Register a root source in the current thread-local root scanner.
pub fn gc_register_root(root: Box<dyn GcRoot>) -> Result<RootId> {
    let _admission = ROOT_ADMISSION.write();
    register_active_root_thread();
    match ROOT_SCANNER.with(|scanner| scanner.register_root(root)) {
        Ok(root_id) => Ok(root_id),
        Err(err) => {
            unregister_active_root_thread();
            Err(err)
        }
    }
}

pub fn gc_unregister_root(root_id: RootId) -> Result<()> {
    let removed = {
        let _admission = ROOT_ADMISSION.write();
        let removed = ROOT_SCANNER.with(|scanner| scanner.take_root(root_id))?;
        unregister_active_root_thread();
        removed
    };
    // Root-owned resources may reenter GC during destruction.
    drop(removed);
    root_retirement::notify(RootRetirement::Registration(root_id));
    Ok(())
}

/// Register a value-visible root under the same lifetime and removal authority
/// as all other GC roots.
pub fn gc_register_value_root(root: Box<dyn ValueRoot>) -> Result<RootId> {
    let _admission = ROOT_ADMISSION.write();
    register_active_root_thread();
    match ROOT_SCANNER.with(|scanner| scanner.register_value_root(root)) {
        Ok(root_id) => Ok(root_id),
        Err(error) => {
            unregister_active_root_thread();
            Err(error)
        }
    }
}

/// Own one value-visible root registration, rather than a handle identity.
/// The registration ID may be excluded from a last-owner inspection while an
/// independently registered root for the same handle remains visible.
#[derive(Debug)]
pub struct ValueRootRegistration {
    id: Option<RootId>,
    // RootScanner is thread-local: this token must be removed on its owner thread.
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl ValueRootRegistration {
    pub fn new(root: Box<dyn ValueRoot>) -> Result<Self> {
        Ok(Self {
            id: Some(gc_register_value_root(root)?),
            _same_thread: std::marker::PhantomData,
        })
    }

    pub fn id(&self) -> RootId {
        self.id.expect("value root registration already removed")
    }

    pub fn unregister(mut self) -> Result<()> {
        let id = self.id.expect("value root registration already removed");
        gc_unregister_root(id)?;
        self.id = None;
        Ok(())
    }
}

impl Drop for ValueRootRegistration {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let _ = gc_unregister_root(id);
        }
    }
}

/// Registered current-thread value roots and explicit-root targets, or
/// unavailable for opaque roots. External domain root providers are not Value
/// registrations; callers must also compose those domains' retained-value views.
/// Project retained Values except registrations owned by a retiring frame.
/// Exclusion affects neither GC tracing nor explicit roots and never unregisters
/// anything; callers must own the excluded registration lifetimes.
pub fn gc_visit_retained_values(
    excluded: &[RootId],
    visitor: &mut dyn FnMut(&Value) -> Option<()>,
) -> Option<()> {
    if registered_roots_exist_on_other_threads() || external_roots_exist_on_other_threads() {
        return None;
    }
    ROOT_SCANNER.with(|scanner| scanner.visit_retained_values(excluded, visitor))?;
    // Snapshot identities before guarded reads; never hold the root table lock
    // across value traversal or invoke callbacks while holding it.
    let explicit = GC.root_ptrs.try_lock()?.keys().copied().collect::<Vec<_>>();
    for handle in explicit {
        let kind = GC.allocator.lock().allocation_kind(&handle)?;
        match kind {
            GcAllocationKind::Value => gc_with_value(&handle, &mut *visitor).ok()??,
            GcAllocationKind::LexicalActivation => {
                gc_with_lexical_activation(&handle, |activation| {
                    let mut accepted = Some(());
                    activation.for_each_value(|value| {
                        if accepted.is_some() {
                            accepted = visitor(value);
                        }
                    });
                    accepted
                })
                .ok()??;
            }
        }
    }
    Some(())
}

/// Trace the complete GC handle graph from registered value roots, counted
/// explicit roots, and installed external root providers. Exclusions are root
/// *registration IDs*, never handle identities: an independent alias of the
/// same target still makes it reachable. `None` means inspection was
/// incomplete and a caller must retain/retry rather than finalize early.
pub fn gc_is_reachable_excluding(target: GcHandle, excluded: &[RootId]) -> Option<bool> {
    gc_with_reachability_excluding(target, excluded, |result| result)
}

/// Hold root-publication admission through the caller's already-prepared
/// Active-to-Queued decision. The closure must not allocate a new GC root,
/// perform user code, or await; it may only update pre-rooted state.
pub fn gc_with_reachability_excluding<R>(
    target: GcHandle,
    excluded: &[RootId],
    decide: impl FnOnce(Option<bool>) -> R,
) -> R {
    let _admission = ROOT_ADMISSION.read();
    decide(reachability_snapshot_excluding(target, excluded))
}

fn reachability_snapshot_excluding(target: GcHandle, excluded: &[RootId]) -> Option<bool> {
    use std::collections::HashSet;

    if GC.collection_in_progress.load(Ordering::Acquire) {
        return None;
    }
    let mut pending = external_roots();
    pending.extend(GC.root_ptrs.try_lock()?.keys().copied());
    gc_visit_retained_values(excluded, &mut |value| {
        pending.extend(gc_handles_in_value(value));
        Some(())
    })?;
    let mut visited = HashSet::new();
    while let Some(handle) = pending.pop() {
        if handle == target {
            return Some(true);
        }
        if visited.insert(handle) {
            let children = gc_handles_in_allocation(&handle)?;
            pending.extend(children);
        }
    }
    Some(false)
}

#[cfg(test)]
fn gc_retained_values() -> Option<Vec<Value>> {
    let mut values = Vec::new();
    gc_visit_retained_values(&[], &mut |value| {
        values.push(value.clone());
        Some(())
    })?;
    Some(values)
}

/// Record a write for GC barriers (approximate old->young tracking)
pub fn gc_record_write(old: &Value, new: &Value) {
    // Generation-aware barrier: record old->young edges discovered through the
    // new value. `new` may be a plain `Value` that contains GC handles rather
    // than a GC allocation itself.
    let old_ptr = old as *const Value as *const u8;
    let alloc = GC.allocator.lock();
    let old_gen = alloc.logical_generation(old_ptr).unwrap_or(0);
    if old_gen == 0 {
        return;
    }
    for young in gc_handles_in_value(new) {
        let young_ptr = young.addr() as *const u8;
        if alloc.is_live_handle(&young)
            && alloc
                .logical_generation(young_ptr)
                .is_some_and(|new_gen| old_gen > new_gen)
        {
            WRITE_BARRIERS.record_reference(old_ptr, young_ptr);
        }
    }
}

/// Record a write into a GC-owned container whose owning allocation is known.
pub fn gc_record_handle_write(owner: &GcHandle, new: &Value) {
    let owner_ptr = owner.addr() as *const u8;
    let alloc = GC.allocator.lock();
    if !alloc.is_live_handle(owner) {
        return;
    }
    let owner_gen = alloc.logical_generation(owner_ptr).unwrap_or(0);
    if owner_gen == 0 {
        return;
    }
    for young in gc_handles_in_value(new) {
        let young_ptr = young.addr() as *const u8;
        if alloc.is_live_handle(&young)
            && alloc
                .logical_generation(young_ptr)
                .is_some_and(|new_gen| owner_gen > new_gen)
        {
            WRITE_BARRIERS.record_reference(owner_ptr, young_ptr);
        }
    }
}

pub(crate) fn gc_handles_in_allocation(handle: &GcHandle) -> Option<Vec<GcHandle>> {
    let kind = GC.allocator.lock().allocation_kind(handle)?;
    match kind {
        GcAllocationKind::Value => gc_with_value(handle, gc_handles_in_value).ok(),
        GcAllocationKind::LexicalActivation => gc_with_lexical_activation(handle, |activation| {
            let mut children = Vec::new();
            activation.for_each_value(|value| children.extend(gc_handles_in_value(value)));
            children
        })
        .ok(),
    }
}

fn gc_handles_in_value(value: &Value) -> Vec<GcHandle> {
    struct HandleCollector {
        handles: Vec<GcHandle>,
    }

    impl Tracer for HandleCollector {
        fn mark(&mut self, handle: GcHandle) {
            self.handles.push(handle);
        }
    }

    let mut tracer = HandleCollector {
        handles: Vec::new(),
    };
    value.trace(&mut tracer);
    tracer.handles
}

/// Get barrier-derived roots for minor GC
pub fn gc_barrier_minor_roots() -> Vec<GcHandle> {
    let alloc = GC.allocator.lock();
    WRITE_BARRIERS
        .get_minor_gc_roots()
        .into_iter()
        .filter_map(|p| alloc.handle_for_live_ptr(p))
        .collect()
}

/// Force a garbage collection for testing/debugging
#[cfg(any(test, feature = "debug-gc"))]
pub fn gc_force_collect() -> Result<usize> {
    gc_collect_major()
}

/// Reset GC for testing - always available
pub fn gc_reset_for_test() -> Result<()> {
    ROOT_SCANNER.with(|scanner| scanner.clear());
    WRITE_BARRIERS.clear_after_major_gc();

    // Reset statistics
    GC.stats.reset();

    // Reset allocator/collector and roots
    let retired = {
        let config = GcConfig::default();
        *GC.config.write() = config.clone();
        let mut alloc = GC.allocator.lock();
        let live_ptrs = alloc.take_tracked_allocations_for_reset();
        let mut retired = Vec::with_capacity(live_ptrs.len());
        for (ptr, handle, kind) in live_ptrs {
            // SAFETY: these pointers are drained from allocator bookkeeping for
            // initialized typed slots that remain live at whole-GC test reset.
            // Each pointer is deduplicated before it is returned.
            unsafe {
                retired.push(retirement::RetiredValue::take(ptr, handle, kind));
            }
        }
        *alloc = GenerationalAllocator::new(&config);
        let mut coll = GC.collector.lock();
        *coll = MarkSweepCollector::new(&config);
        retired
    };

    {
        let mut roots = GC.root_ptrs.lock();
        roots.clear();
    }
    ACTIVE_ROOT_THREADS.lock().clear();

    // Ensure collection is not in progress
    GC.collection_in_progress.store(false, Ordering::Relaxed);
    GC.allocation_in_progress.store(0, Ordering::Relaxed);
    GC.active_value_borrows.store(0, Ordering::Relaxed);
    GC.active_value_mut_borrow.store(false, Ordering::Relaxed);

    // Configure with default settings
    gc_configure(GcConfig::default())?;
    drop(retired);

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
#[test]
fn collection_epoch_advances_only_after_completed_collection() {
    gc_test_context(|| {
        let rooted = gc_allocate_rooted(Value::Num(1.0)).expect("root fixture");
        let before = gc_collection_epoch();
        gc_with_value(&rooted.handle(), |_| {
            assert_eq!(gc_collect_major().expect("deferred borrow collection"), 0);
            assert_eq!(gc_collection_epoch(), before);
        })
        .expect("guarded fixture borrow");
        gc_collect_major().expect("completed collection");
        assert_eq!(gc_collection_epoch(), before + 1);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::NonNull;

    #[test]
    fn collection_and_reset_release_values_and_finalizers_outside_locks() {
        struct AssertUnlocked;
        impl GcFinalizer for AssertUnlocked {
            fn finalize(&self) {
                assert!(GC.allocator.try_lock().is_some());
                assert!(GC.collector.try_lock().is_some());
                assert!(FINALIZERS.try_lock().is_some());
                assert!(!GC.collection_in_progress.load(Ordering::Acquire));
            }
        }
        for collect in [gc_collect_minor, gc_collect_major] {
            gc_test_context(|| {
                let handle = gc_allocate(Value::Num(1.0)).unwrap();
                gc_register_finalizer(handle, Arc::new(AssertUnlocked)).unwrap();
                assert_eq!(collect().unwrap(), 1);
            });
        }
        gc_test_context(|| {
            let handle = gc_allocate(Value::Num(1.0)).unwrap();
            gc_register_finalizer(handle, Arc::new(AssertUnlocked)).unwrap();
            gc_reset_for_test().unwrap();
        });
    }

    #[test]
    fn foreign_payload_release_can_reenter_allocation_after_collection() {
        use runmat_value::{
            ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignRef, ForeignResourceKey,
            ForeignResourceRelease, ForeignTypeIdentity,
        };
        #[derive(Debug)]
        struct Release(AtomicUsize);
        impl ForeignResourceRelease for Release {
            fn release(&self, _: &ForeignResourceKey) {
                // Assert before entry so a regression fails rather than spinning.
                assert!(!GC.collection_in_progress.load(Ordering::Acquire));
                assert!(GC.allocator.try_lock().is_some());
                let allocated = gc_allocate_rooted(Value::Num(7.0)).unwrap();
                assert!(matches!(
                    gc_clone_value(&allocated.handle()).unwrap(),
                    Value::Num(7.0)
                ));
                allocated.unroot().unwrap();
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        for collect in [gc_collect_minor, gc_collect_major] {
            gc_test_context(|| {
                let release = Arc::new(Release(AtomicUsize::new(0)));
                let value = Value::Foreign(ForeignRef::managed(
                    ForeignResourceKey {
                        host_identity: "gc-retirement-test".into(),
                        handle: 1,
                        generation: 1,
                    },
                    ForeignTypeIdentity {
                        family: "test".into(),
                        name: "resource".into(),
                        version: 1,
                    },
                    ForeignOwnership::Shared,
                    ForeignAffinity::OriginProcess,
                    ForeignLifetime::Session,
                    release.clone(),
                ));
                gc_allocate(value).unwrap();
                assert_eq!(collect().unwrap(), 1);
                assert_eq!(release.0.load(Ordering::Relaxed), 1);
                gc_collect_major().unwrap();
                assert_eq!(release.0.load(Ordering::Relaxed), 1);
            });
        }
    }

    #[test]
    fn collection_releases_only_its_gpu_alias_including_nested_values() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        for collect in [gc_collect_minor, gc_collect_major] {
            for nested in [false, true] {
                gc_test_context(|| {
                    let owner = TestAllocationOwner::new();
                    let allocation = owner
                        .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
                        .unwrap();
                    let alias = allocation.handle.clone();
                    let value = Value::GpuTensor(allocation.handle);
                    let value = if nested {
                        Value::Cell(runmat_value::CellArray::new(vec![value], 1, 1).unwrap())
                    } else {
                        value
                    };
                    let target = gc_allocate_rooted(value).unwrap();
                    assert_eq!(collect().unwrap(), 0);
                    assert_eq!(allocation.probe.retirements(), 0);
                    target.unroot().unwrap();
                    assert_eq!(collect().unwrap(), 1);
                    assert_eq!(allocation.probe.retirements(), 0);
                    allocation
                        .probe
                        .with_data(&alias, |data| assert_eq!(data.len(), 1))
                        .unwrap();
                    drop(alias);
                    assert_eq!(allocation.probe.retirements(), 1);
                });
            }
        }
    }

    thread_local! {
        static TEST_EXTERNAL_ROOT: std::cell::Cell<Option<GcHandle>> = const { std::cell::Cell::new(None) };
    }

    fn test_external_roots() -> Vec<GcHandle> {
        TEST_EXTERNAL_ROOT.with(|root| root.get().into_iter().collect())
    }

    fn test_external_roots_on_other_threads() -> bool {
        false
    }

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
    fn registered_external_root_provider_participates_in_collection() {
        gc_test_context(|| {
            register_external_root_provider(
                "runmat-gc-test-external-root",
                test_external_roots,
                test_external_roots_on_other_threads,
            );
            let handle = gc_allocate(Value::String("external-root".into())).unwrap();
            TEST_EXTERNAL_ROOT.with(|root| root.set(Some(handle)));
            gc_collect_major().unwrap();
            assert_eq!(
                gc_clone_value(&handle).unwrap(),
                Value::String("external-root".into())
            );
            TEST_EXTERNAL_ROOT.with(|root| root.set(None));
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
    fn gc_reset_for_test_clears_current_thread_root_scanner() {
        gc_test_context(|| {
            let root = Box::new(crate::roots::GlobalRoot::new(
                vec![Value::Num(1.0)],
                "test root".to_string(),
            ));
            let _root_id = gc_register_root(root).expect("register root");
            ROOT_SCANNER.with(|scanner| {
                assert_eq!(scanner.stats().registered_roots, 1);
            });

            gc_reset_for_test().expect("reset should clear thread scanner");

            ROOT_SCANNER.with(|scanner| {
                assert_eq!(scanner.stats().registered_roots, 0);
            });
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
    fn collection_skips_when_other_thread_has_registered_roots() {
        use runmat_value::HandleRef;
        use std::sync::mpsc;
        use std::thread;

        gc_test_context(|| {
            let (ready_tx, ready_rx) = mpsc::channel();
            let (collect_done_tx, collect_done_rx) = mpsc::channel();
            let (result_tx, result_rx) = mpsc::channel();

            let worker = thread::spawn(move || {
                let handle = gc_allocate(Value::String("thread-root".to_string().into()))
                    .expect("allocation");
                let root_value = Value::HandleObject(HandleRef {
                    class_name: "ThreadRoot".into(),
                    target: handle,
                    valid: true,
                });
                let root_id =
                    gc_register_root(Box::new(GlobalRoot::new(vec![root_value], "worker".into())))
                        .expect("register worker root");

                ready_tx.send(()).expect("signal root ready");
                collect_done_rx.recv().expect("wait for main collection");

                let still_live = gc_clone_value(&handle).is_ok();
                gc_unregister_root(root_id).expect("unregister worker root");
                result_tx.send(still_live).expect("send liveness result");
            });

            ready_rx.recv().expect("worker root should be registered");
            assert_eq!(
                gc_collect_minor().expect("cross-thread collection should skip"),
                0
            );
            collect_done_tx.send(()).expect("signal collection done");
            assert!(
                result_rx.recv().expect("worker should report liveness"),
                "collection from another thread must not invalidate thread-local roots"
            );
            worker.join().expect("worker thread should complete");
        });
    }

    #[test]
    fn test_root_protection() {
        gc_test_context(|| {
            let protected = gc_allocate(Value::Num(42.0)).expect("allocation failed");
            gc_add_root(protected).expect("root registration failed");

            for i in 0..60 {
                let _ = gc_allocate(Value::String(format!("garbage_{i}").into()));
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
    fn explicit_root_unregisters_on_drop() {
        gc_test_context(|| {
            let protected = gc_allocate(Value::String("protected".to_string().into()))
                .expect("allocation failed");
            let addr = protected.addr();

            {
                let root = gc_root(protected).expect("root guard creation failed");
                assert_eq!(root.handle().addr(), addr);
                assert!(GC.root_ptrs.lock().contains_key(&protected));
            }

            assert!(!GC.root_ptrs.lock().contains_key(&protected));
        });
    }

    #[test]
    fn explicit_root_unroot_removes_registration() {
        gc_test_context(|| {
            let protected = gc_allocate(Value::String("protected".to_string().into()))
                .expect("allocation failed");
            let addr = protected.addr();

            let root = gc_root(protected).expect("root guard creation failed");
            assert!(GC.root_ptrs.lock().contains_key(&protected));

            let handle = root.unroot().expect("explicit unroot failed");
            assert_eq!(handle.addr(), addr);
            assert!(!GC.root_ptrs.lock().contains_key(&protected));
        });
    }

    #[test]
    fn duplicate_explicit_roots_are_counted_independently() {
        gc_test_context(|| {
            let protected = gc_allocate(Value::String("protected".to_string().into()))
                .expect("allocation failed");
            let first = gc_root(protected).expect("first root");
            let second = gc_root(protected).expect("second root");

            assert_eq!(GC.root_ptrs.lock().get(&protected).copied(), Some(2));
            drop(first);
            assert_eq!(GC.root_ptrs.lock().get(&protected).copied(), Some(1));

            gc_collect_minor().expect("minor collection failed");
            assert_eq!(
                gc_clone_value(&protected).expect("second root should keep value alive"),
                Value::String("protected".to_string().into())
            );

            drop(second);
            assert!(!GC.root_ptrs.lock().contains_key(&protected));
        });
    }

    #[test]
    fn rooted_allocation_registers_before_collection() {
        gc_test_context(|| {
            let root = gc_allocate_rooted(Value::String("rooted".to_string().into()))
                .expect("rooted alloc");
            let handle = root.handle();

            assert!(GC.root_ptrs.lock().contains_key(&handle));
            gc_collect_minor().expect("minor collection failed");
            assert_eq!(
                gc_clone_value(&handle).expect("rooted handle should remain live"),
                Value::String("rooted".to_string().into())
            );

            drop(root);
            assert!(!GC.root_ptrs.lock().contains_key(&handle));

            // The admitted allocator participates in automatic collection,
            // while its one prepared root protects the new result throughout.
            struct One(GcHandle);
            impl Trace for One {
                fn trace(&self, tracer: &mut dyn Tracer) {
                    tracer.mark(self.0);
                }
            }
            let mut config = gc_get_config();
            config.minor_gc_threshold = 0.0;
            gc_configure(config).expect("automatic collection configuration");
            // The ordinary rooted API protects a child carried only by its
            // incoming Value before honoring the automatic trigger.
            let child = gc_allocate(Value::Num(5.0)).expect("unrooted child");
            let before = gc_collection_epoch();
            let ordinary = gc_allocate_rooted(Value::HandleObject(runmat_value::HandleRef {
                class_name: "RootedAllocationChild".into(),
                target: child,
                valid: true,
            }))
            .expect("ordinary rooted parent");
            assert!(gc_collection_epoch() > before);
            assert_eq!(gc_clone_value(&child).unwrap(), Value::Num(5.0));
            assert!(GC.root_ptrs.lock().contains_key(&ordinary.handle()));

            let before = gc_collection_epoch();
            let activation = gc_allocate_lexical_activation_rooted(LexicalActivation::empty())
                .expect("rooted lexical activation");
            assert!(gc_collection_epoch() > before);
            gc_with_lexical_activation(&activation.handle(), |value| {
                assert_eq!(value.owned_bindings().count(), 0);
            })
            .expect("the exact typed root survives collection");

            let before = gc_collection_epoch();
            let admitted = gc_allocate_prepared_rooted(Value::Num(9.0), |handle| {
                let source = One(handle);
                PreparedRootPlan::for_traced(&source)?.prepare_unbounded(&source)
            })
            .expect("prepared rooted allocation");
            assert!(
                gc_collection_epoch() > before,
                "allocation threshold must collect"
            );
            assert_eq!(gc_clone_value(&admitted.handle()).unwrap(), Value::Num(9.0));
            assert!(GC.root_ptrs.lock().contains_key(&admitted.handle()));
            let handle = admitted.handle();
            drop(admitted);
            assert!(!GC.root_ptrs.lock().contains_key(&handle));
        });
    }

    #[test]
    fn reconfigure_preserves_live_allocations() {
        gc_test_context(|| {
            let root = gc_allocate_rooted(Value::String("still live".to_string().into()))
                .expect("rooted alloc");
            let handle = root.handle();
            let mut config = gc_get_config();
            config.minor_gc_threshold = 0.5;

            gc_configure(config).expect("reconfigure GC");

            assert_eq!(
                gc_clone_value(&handle).expect("reconfiguration must preserve the live heap"),
                Value::String("still live".to_string().into())
            );
        });
    }

    #[test]
    fn reconfigure_can_change_generation_count_before_first_allocation() {
        gc_test_context(|| {
            let mut config = gc_get_config();
            config.num_generations = 2;

            gc_configure(config.clone()).expect("configure empty GC");

            assert_eq!(gc_get_config().num_generations, config.num_generations);
        });
    }

    #[test]
    fn reconfigure_rejects_generation_count_change_with_live_allocations() {
        gc_test_context(|| {
            let root = gc_allocate_rooted(Value::String("still live".to_string().into()))
                .expect("rooted alloc");
            let handle = root.handle();
            let mut config = gc_get_config();
            config.num_generations = 2;

            let error = gc_configure(config).expect_err("live heap must prevent topology change");

            assert!(error.to_string().contains("while GC allocations are live"));
            assert_eq!(
                gc_clone_value(&handle).expect("failed reconfiguration must preserve live heap"),
                Value::String("still live".to_string().into())
            );
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

            gc_add_root(v).expect("root add failed");
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
                GcHandle::from_ptr_unchecked(
                    NonNull::new(raw.cast()).expect("non-null test pointer"),
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
    fn read_guard_derefs_value_and_blocks_collection() {
        gc_test_context(|| {
            let ptr = gc_allocate(Value::Num(9.0)).expect("allocation failed");
            let guard = gc_read_value(&ptr).expect("read guard should succeed");

            assert_eq!(*guard, Value::Num(9.0));
            assert_eq!(
                gc_collect_minor().expect("collection should be skipped during guard"),
                0
            );
        });
    }

    #[test]
    fn write_guard_mutates_value_and_is_exclusive() {
        gc_test_context(|| {
            let ptr = gc_allocate(Value::Num(1.0)).expect("allocation failed");
            {
                let mut guard = gc_write_value(&ptr).expect("write guard should succeed");
                *guard = Value::Num(2.0);

                let nested = gc_write_value(&ptr);
                assert!(matches!(nested, Err(GcError::SyncError(_))));
                assert_eq!(
                    gc_collect_minor().expect("collection should be skipped during guard"),
                    0
                );
            }

            assert_eq!(
                gc_clone_value(&ptr).expect("mutated value should remain live"),
                Value::Num(2.0)
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

    #[test]
    fn cross_thread_mutable_borrow_waits_for_active_reader() {
        gc_test_context(|| {
            let root = gc_allocate_rooted(Value::Num(1.0)).expect("rooted allocation");
            let handle = root.handle();
            let read_guard = gc_read_value(&handle).expect("read guard");
            let (tx, rx) = std::sync::mpsc::channel();

            let writer = std::thread::spawn(move || {
                let writer_root =
                    gc_allocate_rooted(Value::Num(1.0)).expect("writer rooted allocation");
                let mut write_guard = gc_write_value(&writer_root.handle())
                    .expect("writer should wait, then acquire");
                *write_guard = Value::Num(2.0);
                tx.send(()).expect("signal write completion");
            });

            assert!(matches!(
                rx.recv_timeout(std::time::Duration::from_millis(50)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            drop(read_guard);
            rx.recv_timeout(std::time::Duration::from_secs(1))
                .expect("writer should proceed after reader drops");
            writer.join().expect("writer thread");
        });
    }
}
