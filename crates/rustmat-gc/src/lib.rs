//! RustMat Generational Garbage Collector
//! 
//! A high-performance generational garbage collector designed specifically for
//! RustMat's dynamic value system. Features optional pointer compression and
//! optimized collection strategies for numerical computing workloads.

use std::sync::Arc;
use parking_lot::RwLock;
use thiserror::Error;
use rustmat_builtins::Value;

pub mod allocator;
pub mod generations;
pub mod collector;
pub mod roots;
pub mod barriers;
pub mod gc_ptr;
pub mod stats;
pub mod config;

#[cfg(feature = "pointer-compression")]
pub mod compression;

pub use allocator::{GenerationalAllocator, SizeClass, AllocatorStats};
pub use generations::{Generation, GenerationalHeap, GenerationStats, GenerationalHeapStats};
pub use collector::*;
pub use roots::*;
pub use barriers::*;
pub use gc_ptr::*;
pub use stats::*;
pub use config::*;

#[cfg(feature = "pointer-compression")]
pub use compression::*;

/// Global garbage collector instance
static GC: once_cell::sync::Lazy<Arc<RwLock<GarbageCollector>>> = 
    once_cell::sync::Lazy::new(|| {
        Arc::new(RwLock::new(GarbageCollector::new()))
    });

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
}

pub type Result<T> = std::result::Result<T, GcError>;

/// Main garbage collector interface
pub struct GarbageCollector {
    allocator: GenerationalAllocator,
    collector: MarkSweepCollector,
    root_scanner: RootScanner,
    config: GcConfig,
    stats: GcStats,
}

impl GarbageCollector {
    /// Create a new garbage collector with default configuration
    pub fn new() -> Self {
        let config = GcConfig::default();
        Self::with_config(config)
    }
    
    /// Create a new garbage collector with custom configuration
    pub fn with_config(config: GcConfig) -> Self {
        let allocator = GenerationalAllocator::new(&config);
        let collector = MarkSweepCollector::new(&config);
        let root_scanner = RootScanner::new();
        let stats = GcStats::new();
        
        Self {
            allocator,
            collector,
            root_scanner,
            config,
            stats,
        }
    }
    
    /// Allocate a new Value object in the garbage collected heap
    pub fn allocate(&mut self, value: Value) -> Result<GcPtr<Value>> {
        log::trace!("Allocating value: {value:?}");
        
        let ptr = self.allocator.allocate(value, &mut self.stats)?;
        
        // Trigger collection if allocation threshold is reached
        if self.should_collect_minor() {
            self.collect_minor()?;
        }
        
        Ok(ptr)
    }
    
    /// Perform a minor garbage collection (young generation only)
    pub fn collect_minor(&mut self) -> Result<usize> {
        log::debug!("Starting minor garbage collection");
        let start = std::time::Instant::now();
        
        let roots = self.root_scanner.scan_roots()?;
        let collected = self.collector.collect_young_generation(
            &mut self.allocator,
            &roots,
            &mut self.stats
        )?;
        
        let duration = start.elapsed();
        self.stats.record_minor_collection(collected, duration);
        
        log::debug!("Minor collection completed: {collected} objects collected in {duration:?}");
        Ok(collected)
    }
    
    /// Perform a major garbage collection (all generations)
    pub fn collect_major(&mut self) -> Result<usize> {
        log::debug!("Starting major garbage collection");
        let start = std::time::Instant::now();
        
        let roots = self.root_scanner.scan_roots()?;
        let collected = self.collector.collect_all_generations(
            &mut self.allocator,
            &roots,
            &mut self.stats
        )?;
        
        let duration = start.elapsed();
        self.stats.record_major_collection(collected, duration);
        
        log::debug!("Major collection completed: {collected} objects collected in {duration:?}");
        Ok(collected)
    }
    
    /// Register a new GC root (e.g., interpreter stack, variable array)
    pub fn register_root(&mut self, root: Box<dyn GcRoot>) -> Result<RootId> {
        self.root_scanner.register_root(root)
    }
    
    /// Unregister a GC root
    pub fn unregister_root(&mut self, root_id: RootId) -> Result<()> {
        self.root_scanner.unregister_root(root_id)
    }
    
    /// Get current GC statistics
    pub fn stats(&self) -> &GcStats {
        &self.stats
    }
    
    /// Update GC configuration
    pub fn configure(&mut self, config: GcConfig) -> Result<()> {
        log::info!("Updating GC configuration: {config:?}");
        self.config = config;
        self.allocator.reconfigure(&self.config)?;
        self.collector.reconfigure(&self.config)?;
        Ok(())
    }
    
    /// Check if minor collection should be triggered
    fn should_collect_minor(&self) -> bool {
        self.allocator.young_generation_usage() > self.config.minor_gc_threshold
    }
    
    /// Check if major collection should be triggered
    #[allow(dead_code)]
    fn should_collect_major(&self) -> bool {
        self.allocator.total_usage() > self.config.major_gc_threshold
    }
}

impl Default for GarbageCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Global GC functions for easy access
pub fn gc_allocate(value: Value) -> Result<GcPtr<Value>> {
    GC.write().allocate(value)
}

pub fn gc_collect_minor() -> Result<usize> {
    GC.write().collect_minor()
}

pub fn gc_collect_major() -> Result<usize> {
    GC.write().collect_major()
}

pub fn gc_register_root(root: Box<dyn GcRoot>) -> Result<RootId> {
    GC.write().register_root(root)
}

pub fn gc_unregister_root(root_id: RootId) -> Result<()> {
    GC.write().unregister_root(root_id)
}

pub fn gc_stats() -> GcStats {
    GC.read().stats().clone()
}

pub fn gc_configure(config: GcConfig) -> Result<()> {
    GC.write().configure(config)
}

/// Force a garbage collection for testing/debugging
#[cfg(any(test, feature = "debug-gc"))]
pub fn gc_force_collect() -> Result<usize> {
    gc_collect_major()
}



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gc_basic_allocation() {
        let value = Value::Num(42.0);
        let ptr = gc_allocate(value).expect("allocation failed");
        assert_eq!(*ptr, Value::Num(42.0));
    }
    
    #[test]
    fn test_gc_collection() {
        // Allocate many objects to trigger collection
        for i in 0..1000 {
            let _ = gc_allocate(Value::Num(i as f64)).expect("allocation failed");
        }
        
        let collected = gc_collect_minor().expect("collection failed");
        // Most objects should be collected since we don't hold references
        assert!(collected > 0);
    }
}