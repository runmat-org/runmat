//! Garbage collector configuration
//! 
//! Provides configuration options for tuning GC behavior for different
//! workloads and memory constraints.

use std::time::Duration;

/// Configuration for the garbage collector
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Number of generations (minimum 2)
    pub num_generations: usize,
    
    /// Size of the young generation in bytes
    pub young_generation_size: usize,
    
    /// Threshold for triggering minor GC (0.0-1.0, as fraction of young gen)
    pub minor_gc_threshold: f64,
    
    /// Threshold for triggering major GC (0.0-1.0, as fraction of total heap)
    pub major_gc_threshold: f64,
    
    /// Number of collections before promoting objects to next generation
    pub promotion_threshold: usize,
    
    /// Maximum number of GC threads
    pub max_gc_threads: usize,
    
    /// Enable concurrent collection
    pub concurrent_collection: bool,
    
    /// Enable parallel collection
    pub parallel_collection: bool,
    
    /// Collection timeout
    pub collection_timeout: Duration,
    
    /// Enable write barriers for generational collection
    pub write_barriers: bool,
    
    /// Enable pointer compression
    pub pointer_compression: bool,
    
    /// Target heap utilization (0.0-1.0)
    pub target_utilization: f64,
    
    /// Minimum heap size in bytes
    pub min_heap_size: usize,
    
    /// Maximum heap size in bytes (0 = unlimited)
    pub max_heap_size: usize,
    
    /// Growth factor for heap expansion
    pub heap_growth_factor: f64,
    
    /// Enable detailed GC logging
    pub verbose_logging: bool,
    
    /// Enable GC statistics collection
    pub collect_statistics: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            num_generations: 3,
            young_generation_size: 2 * 1024 * 1024, // 2MB
            minor_gc_threshold: 0.8,
            major_gc_threshold: 0.9,
            promotion_threshold: 2,
            max_gc_threads: num_cpus::get(),
            concurrent_collection: false, // Start with simpler implementation
            parallel_collection: false,
            collection_timeout: Duration::from_secs(30),
            write_barriers: true,
            pointer_compression: cfg!(feature = "pointer-compression"),
            target_utilization: 0.7,
            min_heap_size: 1024 * 1024, // 1MB
            max_heap_size: 0, // Unlimited
            heap_growth_factor: 1.5,
            verbose_logging: false,
            collect_statistics: true,
        }
    }
}

impl GcConfig {
    /// Create a configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            minor_gc_threshold: 0.6, // Collect more frequently
            major_gc_threshold: 0.7,
            promotion_threshold: 1, // Promote quickly
            concurrent_collection: true,
            parallel_collection: true,
            target_utilization: 0.5, // Keep heap less full
            ..Default::default()
        }
    }
    
    /// Create a configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            young_generation_size: 64 * 1024 * 1024, // 64MB for high throughput
            minor_gc_threshold: 0.9, // Collect less frequently
            major_gc_threshold: 0.95,
            promotion_threshold: 3, // Keep objects young longer
            parallel_collection: true, // Use parallel collection for throughput
            target_utilization: 0.8, // Allow higher heap usage
            heap_growth_factor: 2.0, // Grow heap more aggressively
            ..Default::default()
        }
    }
    
    /// Create a configuration optimized for memory-constrained environments
    pub fn low_memory() -> Self {
        Self {
            num_generations: 2, // Fewer generations
            young_generation_size: 512 * 1024, // 512KB
            minor_gc_threshold: 0.7,
            major_gc_threshold: 0.8,
            promotion_threshold: 1, // Promote quickly to free young gen
            target_utilization: 0.9, // Use most of available memory
            heap_growth_factor: 1.2, // Conservative growth
            max_heap_size: 16 * 1024 * 1024, // 16MB limit
            ..Default::default()
        }
    }
    
    /// Create a debug configuration with extensive logging and validation
    pub fn debug() -> Self {
        Self {
            verbose_logging: true,
            collect_statistics: true,
            collection_timeout: Duration::from_secs(5), // Shorter timeout for testing
            minor_gc_threshold: 0.5, // Trigger collections frequently
            ..Default::default()
        }
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_generations < 2 {
            return Err("Must have at least 2 generations".to_string());
        }
        
        if self.young_generation_size == 0 {
            return Err("Young generation size must be > 0".to_string());
        }
        
        if !(0.0..=1.0).contains(&self.minor_gc_threshold) {
            return Err("Minor GC threshold must be between 0.0 and 1.0".to_string());
        }
        
        if !(0.0..=1.0).contains(&self.major_gc_threshold) {
            return Err("Major GC threshold must be between 0.0 and 1.0".to_string());
        }
        
        if self.major_gc_threshold < self.minor_gc_threshold {
            return Err("Major GC threshold must be >= minor GC threshold".to_string());
        }
        
        if self.promotion_threshold == 0 {
            return Err("Promotion threshold must be > 0".to_string());
        }
        
        if self.max_gc_threads == 0 {
            return Err("Must have at least 1 GC thread".to_string());
        }
        
        if !(0.0..=1.0).contains(&self.target_utilization) {
            return Err("Target utilization must be between 0.0 and 1.0".to_string());
        }
        
        if self.heap_growth_factor <= 1.0 {
            return Err("Heap growth factor must be > 1.0".to_string());
        }
        
        if self.max_heap_size > 0 && self.max_heap_size < self.min_heap_size {
            return Err("Max heap size must be >= min heap size".to_string());
        }
        
        Ok(())
    }
    
    /// Get total heap capacity across all generations
    pub fn total_heap_capacity(&self) -> usize {
        let mut total = self.young_generation_size;
        let mut gen_size = self.young_generation_size;
        
        for _ in 1..self.num_generations {
            gen_size *= 2;
            total += gen_size;
        }
        
        if self.max_heap_size > 0 {
            total.min(self.max_heap_size)
        } else {
            total
        }
    }
    
    /// Get the size of a specific generation
    pub fn generation_size(&self, generation: usize) -> usize {
        if generation == 0 {
            self.young_generation_size
        } else {
            self.young_generation_size * (1 << generation)
        }
    }
}

/// Environment variable-based configuration builder
pub struct GcConfigBuilder {
    config: GcConfig,
}

impl GcConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: GcConfig::default(),
        }
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut builder = Self::new();
        
        if let Ok(val) = std::env::var("RUSTMAT_GC_YOUNG_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                builder.config.young_generation_size = size;
            }
        }
        
        if let Ok(val) = std::env::var("RUSTMAT_GC_MINOR_THRESHOLD") {
            if let Ok(threshold) = val.parse::<f64>() {
                builder.config.minor_gc_threshold = threshold;
            }
        }
        
        if let Ok(val) = std::env::var("RUSTMAT_GC_MAJOR_THRESHOLD") {
            if let Ok(threshold) = val.parse::<f64>() {
                builder.config.major_gc_threshold = threshold;
            }
        }
        
        if let Ok(val) = std::env::var("RUSTMAT_GC_MAX_HEAP") {
            if let Ok(size) = val.parse::<usize>() {
                builder.config.max_heap_size = size;
            }
        }
        
        if let Ok(val) = std::env::var("RUSTMAT_GC_VERBOSE") {
            builder.config.verbose_logging = val == "1" || val.to_lowercase() == "true";
        }
        
        if let Ok(val) = std::env::var("RUSTMAT_GC_PARALLEL") {
            builder.config.parallel_collection = val == "1" || val.to_lowercase() == "true";
        }
        
        builder
    }
    
    pub fn young_generation_size(mut self, size: usize) -> Self {
        self.config.young_generation_size = size;
        self
    }
    
    pub fn minor_gc_threshold(mut self, threshold: f64) -> Self {
        self.config.minor_gc_threshold = threshold;
        self
    }
    
    pub fn major_gc_threshold(mut self, threshold: f64) -> Self {
        self.config.major_gc_threshold = threshold;
        self
    }
    
    pub fn verbose_logging(mut self, enable: bool) -> Self {
        self.config.verbose_logging = enable;
        self
    }
    
    pub fn build(self) -> Result<GcConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for GcConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config_valid() {
        let config = GcConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_preset_configs_valid() {
        assert!(GcConfig::low_latency().validate().is_ok());
        assert!(GcConfig::high_throughput().validate().is_ok());
        assert!(GcConfig::low_memory().validate().is_ok());
        assert!(GcConfig::debug().validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = GcConfig::default();
        
        // Test invalid num_generations
        config.num_generations = 1;
        assert!(config.validate().is_err());
        config.num_generations = 3;
        
        // Test invalid thresholds
        config.minor_gc_threshold = 1.5;
        assert!(config.validate().is_err());
        config.minor_gc_threshold = 0.8;
        
        config.major_gc_threshold = 0.5; // Less than minor threshold
        assert!(config.validate().is_err());
        config.major_gc_threshold = 0.9;
        
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_heap_capacity_calculation() {
        let config = GcConfig {
            num_generations: 3,
            young_generation_size: 1024,
            ..Default::default()
        };
        
        // Should be 1024 + 2048 + 4096 = 7168
        assert_eq!(config.total_heap_capacity(), 7168);
        
        // Test generation sizes
        assert_eq!(config.generation_size(0), 1024);
        assert_eq!(config.generation_size(1), 2048);
        assert_eq!(config.generation_size(2), 4096);
    }
    
    #[test]
    fn test_config_builder() {
        let config = GcConfigBuilder::new()
            .young_generation_size(1024)
            .minor_gc_threshold(0.7)
            .verbose_logging(true)
            .build()
            .expect("should build valid config");
        
        assert_eq!(config.young_generation_size, 1024);
        assert_eq!(config.minor_gc_threshold, 0.7);
        assert!(config.verbose_logging);
    }
}