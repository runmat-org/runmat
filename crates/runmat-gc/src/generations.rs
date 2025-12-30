//! Generational heap management
//!
//! Manages the heap layout and organization across multiple generations,
//! handling object aging, promotion, and generation-specific optimizations.

use crate::{GcConfig, GcError, Result};
use runmat_time::Instant;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Represents a single generation in the generational heap
#[derive(Debug)]
pub struct Generation {
    /// Generation number (0 = youngest)
    pub number: usize,

    /// Current size in bytes
    current_size: AtomicUsize,

    /// Maximum size in bytes
    max_size: usize,

    /// Number of collections this generation has experienced
    collection_count: AtomicUsize,

    /// Objects that have survived collections
    survivor_count: AtomicUsize,

    /// Time of last collection
    last_collection: parking_lot::Mutex<Option<Instant>>,

    /// Age threshold for promotion to next generation
    promotion_threshold: usize,

    /// Collection frequency (for adaptive scheduling)
    collection_frequency: parking_lot::Mutex<VecDeque<Instant>>,
}

impl Generation {
    pub fn new(number: usize, max_size: usize, promotion_threshold: usize) -> Self {
        Self {
            number,
            current_size: AtomicUsize::new(0),
            max_size,
            collection_count: AtomicUsize::new(0),
            survivor_count: AtomicUsize::new(0),
            last_collection: parking_lot::Mutex::new(None),
            promotion_threshold,
            collection_frequency: parking_lot::Mutex::new(VecDeque::new()),
        }
    }

    /// Get current size
    pub fn current_size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }

    /// Get maximum size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get utilization as a percentage
    pub fn utilization(&self) -> f64 {
        self.current_size() as f64 / self.max_size as f64
    }

    /// Check if generation is full
    pub fn is_full(&self, threshold: f64) -> bool {
        self.utilization() >= threshold
    }

    /// Allocate bytes in this generation
    pub fn allocate(&self, size: usize) -> Result<()> {
        let current = self.current_size.load(Ordering::Relaxed);
        if current + size > self.max_size {
            return Err(GcError::OutOfMemory(format!(
                "Generation {} cannot allocate {} bytes",
                self.number, size
            )));
        }

        self.current_size.fetch_add(size, Ordering::Relaxed);
        Ok(())
    }

    /// Deallocate bytes from this generation
    pub fn deallocate(&self, size: usize) {
        self.current_size
            .fetch_sub(size.min(self.current_size()), Ordering::Relaxed);
    }

    /// Record a collection in this generation
    pub fn record_collection(&self, objects_collected: usize, survivors: usize) {
        self.collection_count.fetch_add(1, Ordering::Relaxed);
        self.survivor_count.store(survivors, Ordering::Relaxed);

        let now = Instant::now();
        *self.last_collection.lock() = Some(now);

        // Track collection frequency
        let mut frequency = self.collection_frequency.lock();
        frequency.push_back(now);

        // Keep only recent collections (last 60 seconds)
        let cutoff = now - Duration::from_secs(60);
        while frequency.front().is_some_and(|&t| t < cutoff) {
            frequency.pop_front();
        }

        log::debug!(
            "Generation {} collection: {} collected, {} survivors",
            self.number,
            objects_collected,
            survivors
        );
    }

    /// Get collection count
    pub fn collection_count(&self) -> usize {
        self.collection_count.load(Ordering::Relaxed)
    }

    /// Get survivor count from last collection
    pub fn survivor_count(&self) -> usize {
        self.survivor_count.load(Ordering::Relaxed)
    }

    /// Check if objects should be promoted based on survival count
    pub fn should_promote(&self, object_age: usize) -> bool {
        object_age >= self.promotion_threshold
    }

    /// Get time since last collection
    pub fn time_since_last_collection(&self) -> Option<Duration> {
        self.last_collection.lock().map(|time| time.elapsed())
    }

    /// Get collection frequency (collections per minute)
    pub fn collection_frequency(&self) -> f64 {
        let frequency = self.collection_frequency.lock();
        if frequency.len() < 2 {
            return 0.0;
        }

        let duration = frequency
            .back()
            .unwrap()
            .duration_since(*frequency.front().unwrap());
        if duration.as_secs_f64() == 0.0 {
            return 0.0;
        }

        frequency.len() as f64 / (duration.as_secs_f64() / 60.0)
    }

    /// Resize the generation
    pub fn resize(&mut self, new_max_size: usize) -> Result<()> {
        if new_max_size < self.current_size() {
            return Err(GcError::ConfigError(format!(
                "Cannot shrink generation {} below current size",
                self.number
            )));
        }

        log::info!(
            "Resizing generation {} from {} to {} bytes",
            self.number,
            self.max_size,
            new_max_size
        );
        self.max_size = new_max_size;
        Ok(())
    }

    /// Reset generation state (after major collection)
    pub fn reset(&self) {
        self.current_size.store(0, Ordering::Relaxed);
        self.survivor_count.store(0, Ordering::Relaxed);
        // Don't reset collection_count as it's cumulative
    }

    /// Get generation statistics
    pub fn stats(&self) -> GenerationStats {
        GenerationStats {
            number: self.number,
            current_size: self.current_size(),
            max_size: self.max_size,
            utilization: self.utilization(),
            collection_count: self.collection_count(),
            survivor_count: self.survivor_count(),
            promotion_threshold: self.promotion_threshold,
            time_since_last_collection: self.time_since_last_collection(),
            collection_frequency: self.collection_frequency(),
        }
    }
}

/// Statistics for a generation
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub number: usize,
    pub current_size: usize,
    pub max_size: usize,
    pub utilization: f64,
    pub collection_count: usize,
    pub survivor_count: usize,
    pub promotion_threshold: usize,
    pub time_since_last_collection: Option<Duration>,
    pub collection_frequency: f64,
}

/// Manages all generations in the heap
pub struct GenerationalHeap {
    /// All generations (ordered from young to old)
    generations: Vec<Generation>,

    /// Configuration
    config: GcConfig,

    /// Total heap size limit
    total_size_limit: usize,

    /// Adaptive sizing enabled
    adaptive_sizing: bool,

    /// Statistics
    total_promotions: AtomicUsize,
    total_demotions: AtomicUsize,
}

impl GenerationalHeap {
    pub fn new(config: &GcConfig) -> Self {
        let mut generations = Vec::new();

        // Create generations with exponentially increasing sizes
        let mut gen_size = config.young_generation_size;
        for i in 0..config.num_generations {
            generations.push(Generation::new(i, gen_size, config.promotion_threshold));
            gen_size *= 2; // Each generation is twice as large as the previous
        }

        let total_size_limit = if config.max_heap_size > 0 {
            config.max_heap_size
        } else {
            // Calculate total based on generation sizes
            let mut total = config.young_generation_size;
            let mut size = config.young_generation_size;
            for _ in 1..config.num_generations {
                size *= 2;
                total += size;
            }
            total
        };

        Self {
            generations,
            config: config.clone(),
            total_size_limit,
            adaptive_sizing: true,
            total_promotions: AtomicUsize::new(0),
            total_demotions: AtomicUsize::new(0),
        }
    }

    /// Get a specific generation
    pub fn generation(&self, number: usize) -> Option<&Generation> {
        self.generations.get(number)
    }

    /// Get the young generation (generation 0)
    pub fn young_generation(&self) -> &Generation {
        &self.generations[0]
    }

    /// Get all generations
    pub fn generations(&self) -> &[Generation] {
        &self.generations
    }

    /// Get number of generations
    pub fn num_generations(&self) -> usize {
        self.generations.len()
    }

    /// Check if minor GC should be triggered
    pub fn should_collect_minor(&self) -> bool {
        self.young_generation()
            .is_full(self.config.minor_gc_threshold)
    }

    /// Check if major GC should be triggered
    pub fn should_collect_major(&self) -> bool {
        self.total_utilization() >= self.config.major_gc_threshold
    }

    /// Get total heap utilization
    pub fn total_utilization(&self) -> f64 {
        let total_used: usize = self.generations.iter().map(|g| g.current_size()).sum();

        total_used as f64 / self.total_size_limit as f64
    }

    /// Get total current size
    pub fn total_current_size(&self) -> usize {
        self.generations.iter().map(|g| g.current_size()).sum()
    }

    /// Get total maximum size
    pub fn total_max_size(&self) -> usize {
        self.generations.iter().map(|g| g.max_size()).sum()
    }

    /// Record object promotion between generations
    pub fn record_promotion(&self, _from_gen: usize, _to_gen: usize, _count: usize) {
        self.total_promotions.fetch_add(_count, Ordering::Relaxed);

        log::trace!("Promoted {_count} objects from generation {_from_gen} to {_to_gen}");
    }

    /// Record object demotion (rare, but possible in some algorithms)
    pub fn record_demotion(&self, _from_gen: usize, _to_gen: usize, _count: usize) {
        self.total_demotions.fetch_add(_count, Ordering::Relaxed);

        log::trace!("Demoted {_count} objects from generation {_from_gen} to {_to_gen}");
    }

    /// Adapt generation sizes based on allocation patterns
    pub fn adapt_generation_sizes(&mut self) -> Result<()> {
        if !self.adaptive_sizing {
            return Ok(());
        }

        log::debug!("Adapting generation sizes based on allocation patterns");

        // Simple adaptive strategy:
        // - If young generation is collecting too frequently, increase its size
        // - If old generations are mostly empty, shrink them

        let young_gen_freq = self.young_generation().collection_frequency();
        let young_gen_util = self.young_generation().utilization();

        // If collecting more than 10 times per minute and utilization > 80%
        if young_gen_freq > 10.0 && young_gen_util > 0.8 {
            let new_size = (self.generations[0].max_size() as f64 * 1.2) as usize;
            if new_size <= self.total_size_limit / 4 {
                // Don't let young gen be > 25% of total
                log::info!("Increasing young generation size to {new_size} bytes");
                self.generations[0].max_size = new_size;
            }
        }

        // Check old generations for underutilization
        for i in 1..self.generations.len() {
            let gen = &self.generations[i];
            if gen.utilization() < 0.2 && gen.max_size() > self.config.young_generation_size {
                let new_size = (gen.max_size() as f64 * 0.9) as usize;
                log::info!("Decreasing generation {i} size to {new_size} bytes");
                self.generations[i].max_size = new_size.max(self.config.young_generation_size);
            }
        }

        Ok(())
    }

    /// Reconfigure the heap
    pub fn reconfigure(&mut self, config: &GcConfig) -> Result<()> {
        // Validate that we can accommodate the new configuration
        if config.num_generations != self.generations.len() {
            return Err(GcError::ConfigError(
                "Cannot change number of generations at runtime".to_string(),
            ));
        }

        // Update promotion thresholds
        for gen in self.generations.iter_mut() {
            gen.promotion_threshold = config.promotion_threshold;
        }

        self.config = config.clone();

        // Potentially resize young generation
        if config.young_generation_size != self.generations[0].max_size() {
            self.generations[0].resize(config.young_generation_size)?;
        }

        Ok(())
    }

    /// Get comprehensive heap statistics
    pub fn stats(&self) -> GenerationalHeapStats {
        GenerationalHeapStats {
            generations: self.generations.iter().map(|g| g.stats()).collect(),
            total_current_size: self.total_current_size(),
            total_max_size: self.total_max_size(),
            total_utilization: self.total_utilization(),
            total_promotions: self.total_promotions.load(Ordering::Relaxed),
            total_demotions: self.total_demotions.load(Ordering::Relaxed),
            should_collect_minor: self.should_collect_minor(),
            should_collect_major: self.should_collect_major(),
        }
    }
}

/// Comprehensive statistics for the generational heap
#[derive(Debug, Clone)]
pub struct GenerationalHeapStats {
    pub generations: Vec<GenerationStats>,
    pub total_current_size: usize,
    pub total_max_size: usize,
    pub total_utilization: f64,
    pub total_promotions: usize,
    pub total_demotions: usize,
    pub should_collect_minor: bool,
    pub should_collect_major: bool,
}

impl GenerationalHeapStats {
    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = format!(
            "Generational Heap Summary:\n\
             Total Size: {} / {} bytes ({:.1}% utilized)\n\
             Promotions: {}, Demotions: {}\n\
             Collection Triggers: Minor={}, Major={}\n\n",
            self.total_current_size,
            self.total_max_size,
            self.total_utilization * 100.0,
            self.total_promotions,
            self.total_demotions,
            self.should_collect_minor,
            self.should_collect_major
        );

        for gen in &self.generations {
            report.push_str(&format!(
                "Generation {}: {} / {} bytes ({:.1}% util), {} collections, freq={:.1}/min\n",
                gen.number,
                gen.current_size,
                gen.max_size,
                gen.utilization * 100.0,
                gen.collection_count,
                gen.collection_frequency
            ));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GcConfig;

    #[test]
    fn test_generation_basic() {
        let gen = Generation::new(0, 1024, 2);

        assert_eq!(gen.number, 0);
        assert_eq!(gen.max_size(), 1024);
        assert_eq!(gen.current_size(), 0);
        assert_eq!(gen.utilization(), 0.0);
        assert!(!gen.is_full(0.8));

        // Test allocation
        gen.allocate(512).expect("should allocate");
        assert_eq!(gen.current_size(), 512);
        assert_eq!(gen.utilization(), 0.5);

        // Test deallocation
        gen.deallocate(256);
        assert_eq!(gen.current_size(), 256);
    }

    #[test]
    fn test_generation_promotion() {
        let gen = Generation::new(0, 1024, 2);

        assert!(!gen.should_promote(1));
        assert!(gen.should_promote(2));
        assert!(gen.should_promote(3));
    }

    #[test]
    fn test_generation_collection_tracking() {
        let gen = Generation::new(0, 1024, 2);

        assert_eq!(gen.collection_count(), 0);
        assert_eq!(gen.survivor_count(), 0);

        gen.record_collection(10, 5);
        assert_eq!(gen.collection_count(), 1);
        assert_eq!(gen.survivor_count(), 5);

        gen.record_collection(8, 3);
        assert_eq!(gen.collection_count(), 2);
        assert_eq!(gen.survivor_count(), 3);
    }

    #[test]
    fn test_generational_heap() {
        let config = GcConfig::default();
        let heap = GenerationalHeap::new(&config);

        assert_eq!(heap.num_generations(), config.num_generations);
        assert_eq!(heap.young_generation().number, 0);

        // Test collection triggers
        assert!(!heap.should_collect_minor()); // Empty heap
        assert!(!heap.should_collect_major());

        // Fill young generation partially
        heap.young_generation()
            .allocate((config.young_generation_size as f64 * 0.9) as usize)
            .expect("should allocate");

        assert!(heap.should_collect_minor()); // Should trigger minor GC
    }

    #[test]
    fn test_heap_statistics() {
        let config = GcConfig::default();
        let heap = GenerationalHeap::new(&config);

        let stats = heap.stats();
        assert_eq!(stats.generations.len(), config.num_generations);
        assert_eq!(stats.total_current_size, 0);
        assert!(stats.total_max_size > 0);
        assert_eq!(stats.total_utilization, 0.0);

        // Allocate some memory
        heap.young_generation()
            .allocate(1000)
            .expect("should allocate");

        let stats = heap.stats();
        assert_eq!(stats.total_current_size, 1000);
        assert!(stats.total_utilization > 0.0);
    }

    #[test]
    fn test_generation_resize() {
        let mut gen = Generation::new(0, 1024, 2);

        // Resize up
        gen.resize(2048).expect("should resize");
        assert_eq!(gen.max_size(), 2048);

        // Allocate some memory
        gen.allocate(1500).expect("should allocate");

        // Can't resize below current usage
        assert!(gen.resize(1000).is_err());

        // Can resize to accommodate current usage
        assert!(gen.resize(1500).is_ok());
    }

    #[test]
    fn test_heap_total_utilization() {
        let config = GcConfig::default();
        let heap = GenerationalHeap::new(&config);

        // Allocate in multiple generations
        heap.generation(0)
            .unwrap()
            .allocate(1000)
            .expect("should allocate");
        heap.generation(1)
            .unwrap()
            .allocate(2000)
            .expect("should allocate");

        let utilization = heap.total_utilization();
        assert!(utilization > 0.0);
        assert!(utilization < 1.0);

        let total_used = heap.total_current_size();
        assert_eq!(total_used, 3000);
    }
}
