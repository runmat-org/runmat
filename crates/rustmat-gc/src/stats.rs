//! Garbage collection statistics and metrics
//! 
//! Provides detailed statistics about GC performance, memory usage,
//! and collection behavior for monitoring and tuning.

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::collections::VecDeque;

/// Comprehensive GC statistics
#[derive(Debug)]
pub struct GcStats {
    /// Total number of allocations
    pub total_allocations: AtomicUsize,
    
    /// Total bytes allocated
    pub total_allocated_bytes: AtomicU64,
    
    /// Number of minor collections performed
    pub minor_collections: AtomicUsize,
    
    /// Number of major collections performed  
    pub major_collections: AtomicUsize,
    
    /// Total time spent in minor collections
    pub minor_collection_time: AtomicU64,
    
    /// Total time spent in major collections
    pub major_collection_time: AtomicU64,
    
    /// Objects collected in minor collections
    pub minor_objects_collected: AtomicUsize,
    
    /// Objects collected in major collections
    pub major_objects_collected: AtomicUsize,
    
    /// Objects promoted between generations
    pub objects_promoted: AtomicUsize,
    
    /// Peak memory usage
    pub peak_memory_usage: AtomicUsize,
    
    /// Current memory usage
    pub current_memory_usage: AtomicUsize,
    
    /// Collection history for trend analysis
    collection_history: parking_lot::Mutex<VecDeque<CollectionEvent>>,
    
    /// Allocation rate tracking
    allocation_timestamps: parking_lot::Mutex<VecDeque<Instant>>,
    
    /// Start time for rate calculations
    start_time: Instant,
}

/// Information about a single garbage collection event
#[derive(Debug, Clone)]
pub struct CollectionEvent {
    pub timestamp: Instant,
    pub collection_type: CollectionType,
    pub duration: Duration,
    pub objects_collected: usize,
    pub bytes_collected: usize,
    pub heap_size_before: usize,
    pub heap_size_after: usize,
    pub promotion_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionType {
    Minor,
    Major,
}

impl GcStats {
    pub fn new() -> Self {
        Self {
            total_allocations: AtomicUsize::new(0),
            total_allocated_bytes: AtomicU64::new(0),
            minor_collections: AtomicUsize::new(0),
            major_collections: AtomicUsize::new(0),
            minor_collection_time: AtomicU64::new(0),
            major_collection_time: AtomicU64::new(0),
            minor_objects_collected: AtomicUsize::new(0),
            major_objects_collected: AtomicUsize::new(0),
            objects_promoted: AtomicUsize::new(0),
            peak_memory_usage: AtomicUsize::new(0),
            current_memory_usage: AtomicUsize::new(0),
            collection_history: parking_lot::Mutex::new(VecDeque::new()),
            allocation_timestamps: parking_lot::Mutex::new(VecDeque::new()),
            start_time: Instant::now(),
        }
    }
    
    /// Record an allocation
    pub fn record_allocation(&self, size: usize) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_allocated_bytes.fetch_add(size as u64, Ordering::Relaxed);
        
        let new_usage = self.current_memory_usage.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak usage
        let mut peak = self.peak_memory_usage.load(Ordering::Relaxed);
        while new_usage > peak {
            match self.peak_memory_usage.compare_exchange_weak(
                peak, new_usage, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
        
        // Track allocation rate
        let mut timestamps = self.allocation_timestamps.lock();
        timestamps.push_back(Instant::now());
        
        // Keep only recent allocations for rate calculation (last 60 seconds)
        let cutoff = Instant::now() - Duration::from_secs(60);
        while timestamps.front().is_some_and(|&t| t < cutoff) {
            timestamps.pop_front();
        }
    }
    
    /// Record a minor collection
    pub fn record_minor_collection(&self, objects_collected: usize, duration: Duration) {
        self.minor_collections.fetch_add(1, Ordering::Relaxed);
        self.minor_objects_collected.fetch_add(objects_collected, Ordering::Relaxed);
        self.minor_collection_time.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        
        self.record_collection_event(CollectionEvent {
            timestamp: Instant::now(),
            collection_type: CollectionType::Minor,
            duration,
            objects_collected,
            bytes_collected: 0, // TODO: Track bytes collected
            heap_size_before: 0,
            heap_size_after: 0,
            promotion_count: 0,
        });
    }
    
    /// Record a major collection
    pub fn record_major_collection(&self, objects_collected: usize, duration: Duration) {
        self.major_collections.fetch_add(1, Ordering::Relaxed);
        self.major_objects_collected.fetch_add(objects_collected, Ordering::Relaxed);
        self.major_collection_time.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        
        self.record_collection_event(CollectionEvent {
            timestamp: Instant::now(),
            collection_type: CollectionType::Major,
            duration,
            objects_collected,
            bytes_collected: 0,
            heap_size_before: 0,
            heap_size_after: 0,
            promotion_count: 0,
        });
    }
    
    /// Record object promotion
    pub fn record_promotion(&self, count: usize) {
        self.objects_promoted.fetch_add(count, Ordering::Relaxed);
    }
    
    /// Record memory deallocation
    pub fn record_deallocation(&self, size: usize) {
        self.current_memory_usage.fetch_sub(size, Ordering::Relaxed);
    }
    
    /// Get current allocation rate (allocations per second)
    pub fn allocation_rate(&self) -> f64 {
        let timestamps = self.allocation_timestamps.lock();
        if timestamps.len() < 2 {
            return 0.0;
        }
        
        let duration = timestamps.back().unwrap().duration_since(*timestamps.front().unwrap());
        if duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        
        timestamps.len() as f64 / duration.as_secs_f64()
    }
    
    /// Get average minor collection time
    pub fn average_minor_collection_time(&self) -> Duration {
        let total_time = Duration::from_nanos(self.minor_collection_time.load(Ordering::Relaxed));
        let count = self.minor_collections.load(Ordering::Relaxed);
        
        if count == 0 {
            Duration::ZERO
        } else {
            total_time / count as u32
        }
    }
    
    /// Get average major collection time
    pub fn average_major_collection_time(&self) -> Duration {
        let total_time = Duration::from_nanos(self.major_collection_time.load(Ordering::Relaxed));
        let count = self.major_collections.load(Ordering::Relaxed);
        
        if count == 0 {
            Duration::ZERO
        } else {
            total_time / count as u32
        }
    }
    
    /// Get GC overhead as percentage of total runtime
    pub fn gc_overhead_percentage(&self) -> f64 {
        let total_gc_time = Duration::from_nanos(
            self.minor_collection_time.load(Ordering::Relaxed) +
            self.major_collection_time.load(Ordering::Relaxed)
        );
        
        let total_runtime = self.start_time.elapsed();
        
        if total_runtime.as_nanos() == 0 {
            0.0
        } else {
            (total_gc_time.as_nanos() as f64 / total_runtime.as_nanos() as f64) * 100.0
        }
    }
    
    /// Get memory utilization percentage
    pub fn memory_utilization(&self) -> f64 {
        let current = self.current_memory_usage.load(Ordering::Relaxed);
        let peak = self.peak_memory_usage.load(Ordering::Relaxed);
        
        if peak == 0 {
            0.0
        } else {
            (current as f64 / peak as f64) * 100.0
        }
    }
    
    /// Get collection frequency (collections per minute)
    pub fn collection_frequency(&self) -> (f64, f64) {
        let runtime_minutes = self.start_time.elapsed().as_secs_f64() / 60.0;
        if runtime_minutes == 0.0 {
            return (0.0, 0.0);
        }
        
        let minor_freq = self.minor_collections.load(Ordering::Relaxed) as f64 / runtime_minutes;
        let major_freq = self.major_collections.load(Ordering::Relaxed) as f64 / runtime_minutes;
        
        (minor_freq, major_freq)
    }
    
    /// Get recent collection history
    pub fn recent_collections(&self, limit: usize) -> Vec<CollectionEvent> {
        let history = self.collection_history.lock();
        history.iter().rev().take(limit).cloned().collect()
    }
    
    /// Record a collection event
    fn record_collection_event(&self, event: CollectionEvent) {
        let mut history = self.collection_history.lock();
        history.push_back(event);
        
        // Keep only recent history (last 1000 events)
        while history.len() > 1000 {
            history.pop_front();
        }
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        self.total_allocations.store(0, Ordering::Relaxed);
        self.total_allocated_bytes.store(0, Ordering::Relaxed);
        self.minor_collections.store(0, Ordering::Relaxed);
        self.major_collections.store(0, Ordering::Relaxed);
        self.minor_collection_time.store(0, Ordering::Relaxed);
        self.major_collection_time.store(0, Ordering::Relaxed);
        self.minor_objects_collected.store(0, Ordering::Relaxed);
        self.major_objects_collected.store(0, Ordering::Relaxed);
        self.objects_promoted.store(0, Ordering::Relaxed);
        self.peak_memory_usage.store(0, Ordering::Relaxed);
        self.current_memory_usage.store(0, Ordering::Relaxed);
        
        self.collection_history.lock().clear();
        self.allocation_timestamps.lock().clear();
    }
    
    /// Generate a summary report
    pub fn summary_report(&self) -> String {
        let total_allocs = self.total_allocations.load(Ordering::Relaxed);
        let total_bytes = self.total_allocated_bytes.load(Ordering::Relaxed);
        let minor_colls = self.minor_collections.load(Ordering::Relaxed);
        let major_colls = self.major_collections.load(Ordering::Relaxed);
        let current_mem = self.current_memory_usage.load(Ordering::Relaxed);
        let peak_mem = self.peak_memory_usage.load(Ordering::Relaxed);
        let (minor_freq, major_freq) = self.collection_frequency();
        
        format!(
            "GC Statistics Summary:\n\
             Allocations: {} ({} bytes)\n\
             Current Memory: {} bytes (Peak: {} bytes)\n\
             Minor Collections: {} (avg {:.2}ms, {:.1}/min)\n\
             Major Collections: {} (avg {:.2}ms, {:.1}/min)\n\
             GC Overhead: {:.2}%\n\
             Allocation Rate: {:.1} allocs/sec\n\
             Memory Utilization: {:.1}%",
            total_allocs,
            total_bytes,
            current_mem,
            peak_mem,
            minor_colls,
            self.average_minor_collection_time().as_secs_f64() * 1000.0,
            minor_freq,
            major_colls,
            self.average_major_collection_time().as_secs_f64() * 1000.0,
            major_freq,
            self.gc_overhead_percentage(),
            self.allocation_rate(),
            self.memory_utilization()
        )
    }
}

impl Clone for GcStats {
    fn clone(&self) -> Self {
        Self {
            total_allocations: AtomicUsize::new(self.total_allocations.load(Ordering::Relaxed)),
            total_allocated_bytes: AtomicU64::new(self.total_allocated_bytes.load(Ordering::Relaxed)),
            minor_collections: AtomicUsize::new(self.minor_collections.load(Ordering::Relaxed)),
            major_collections: AtomicUsize::new(self.major_collections.load(Ordering::Relaxed)),
            minor_collection_time: AtomicU64::new(self.minor_collection_time.load(Ordering::Relaxed)),
            major_collection_time: AtomicU64::new(self.major_collection_time.load(Ordering::Relaxed)),
            minor_objects_collected: AtomicUsize::new(self.minor_objects_collected.load(Ordering::Relaxed)),
            major_objects_collected: AtomicUsize::new(self.major_objects_collected.load(Ordering::Relaxed)),
            objects_promoted: AtomicUsize::new(self.objects_promoted.load(Ordering::Relaxed)),
            peak_memory_usage: AtomicUsize::new(self.peak_memory_usage.load(Ordering::Relaxed)),
            current_memory_usage: AtomicUsize::new(self.current_memory_usage.load(Ordering::Relaxed)),
            collection_history: parking_lot::Mutex::new(self.collection_history.lock().clone()),
            allocation_timestamps: parking_lot::Mutex::new(self.allocation_timestamps.lock().clone()),
            start_time: self.start_time,
        }
    }
}

impl Default for GcStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for GC tuning
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub allocation_rate: f64,
    pub gc_overhead: f64,
    pub memory_efficiency: f64,
    pub collection_latency: Duration,
    pub throughput: f64,
}

impl PerformanceMetrics {
    pub fn from_stats(stats: &GcStats) -> Self {
        Self {
            allocation_rate: stats.allocation_rate(),
            gc_overhead: stats.gc_overhead_percentage(),
            memory_efficiency: stats.memory_utilization(),
            collection_latency: stats.average_minor_collection_time().max(
                stats.average_major_collection_time()
            ),
            throughput: stats.total_allocations.load(Ordering::Relaxed) as f64 /
                       stats.start_time.elapsed().as_secs_f64(),
        }
    }
    
    /// Get an overall performance score (0-100)
    pub fn performance_score(&self) -> f64 {
        let latency_score = if self.collection_latency.as_millis() < 10 {
            100.0
        } else {
            (100.0 / (self.collection_latency.as_millis() as f64 / 10.0)).min(100.0)
        };
        
        let overhead_score = (100.0 - self.gc_overhead).max(0.0);
        let efficiency_score = self.memory_efficiency;
        
        (latency_score + overhead_score + efficiency_score) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_stats_basic_operations() {
        let stats = GcStats::new();
        
        // Test allocation recording
        stats.record_allocation(100);
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_allocated_bytes.load(Ordering::Relaxed), 100);
        assert_eq!(stats.current_memory_usage.load(Ordering::Relaxed), 100);
        
        // Test deallocation
        stats.record_deallocation(50);
        assert_eq!(stats.current_memory_usage.load(Ordering::Relaxed), 50);
    }
    
    #[test]
    fn test_collection_recording() {
        let stats = GcStats::new();
        
        stats.record_minor_collection(10, Duration::from_millis(5));
        assert_eq!(stats.minor_collections.load(Ordering::Relaxed), 1);
        assert_eq!(stats.minor_objects_collected.load(Ordering::Relaxed), 10);
        
        stats.record_major_collection(50, Duration::from_millis(20));
        assert_eq!(stats.major_collections.load(Ordering::Relaxed), 1);
        assert_eq!(stats.major_objects_collected.load(Ordering::Relaxed), 50);
    }
    
    #[test]
    fn test_allocation_rate() {
        let stats = GcStats::new();
        
        // Record some allocations with time delays
        stats.record_allocation(100);
        thread::sleep(Duration::from_millis(100));
        stats.record_allocation(100);
        thread::sleep(Duration::from_millis(100));
        stats.record_allocation(100);
        
        let rate = stats.allocation_rate();
        assert!(rate > 0.0);
        assert!(rate < 100.0); // Should be reasonable
    }
    
    #[test]
    fn test_performance_metrics() {
        let stats = GcStats::new();
        stats.record_allocation(1000);
        stats.record_minor_collection(5, Duration::from_millis(2));
        
        let metrics = PerformanceMetrics::from_stats(&stats);
        assert!(metrics.performance_score() >= 0.0);
        assert!(metrics.performance_score() <= 100.0);
    }
    
    #[test]
    fn test_stats_reset() {
        let stats = GcStats::new();
        
        stats.record_allocation(100);
        stats.record_minor_collection(5, Duration::from_millis(2));
        
        assert!(stats.total_allocations.load(Ordering::Relaxed) > 0);
        
        stats.reset();
        
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 0);
        assert_eq!(stats.minor_collections.load(Ordering::Relaxed), 0);
    }
}