//! Hotspot Profiler for RustMat Turbine
//!
//! Tracks execution frequency of bytecode sequences to identify hot paths
//! that should be compiled to native code for optimal performance.

use std::collections::HashMap;

/// Tracks execution frequency and identifies hot code paths
pub struct HotspotProfiler {
    execution_counts: HashMap<u64, u32>,
    total_executions: u64,
    hot_threshold: u32,
}

impl HotspotProfiler {
    /// Create a new profiler with default hot threshold
    pub fn new() -> Self {
        Self::with_threshold(10) // Functions become "hot" after 10 executions
    }

    /// Create a profiler with custom hot threshold
    pub fn with_threshold(hot_threshold: u32) -> Self {
        Self {
            execution_counts: HashMap::new(),
            total_executions: 0,
            hot_threshold,
        }
    }

    /// Record an execution of bytecode
    pub fn record_execution(&mut self, bytecode_hash: u64) {
        *self.execution_counts.entry(bytecode_hash).or_insert(0) += 1;
        self.total_executions += 1;
    }

    /// Check if a bytecode sequence is hot enough to compile
    pub fn is_hot(&self, bytecode_hash: u64) -> bool {
        self.execution_counts
            .get(&bytecode_hash)
            .map(|&count| count >= self.hot_threshold)
            .unwrap_or(false)
    }

    /// Get the execution count for a bytecode sequence
    pub fn get_hotness(&self, bytecode_hash: u64) -> u32 {
        self.execution_counts
            .get(&bytecode_hash)
            .copied()
            .unwrap_or(0)
    }

    /// Get total number of executions tracked
    pub fn total_executions(&self) -> u64 {
        self.total_executions
    }

    /// Get the hottest functions (most frequently executed)
    pub fn get_hottest_functions(&self, limit: usize) -> Vec<(u64, u32)> {
        let mut functions: Vec<_> = self
            .execution_counts
            .iter()
            .map(|(&hash, &count)| (hash, count))
            .collect();

        functions.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending
        functions.truncate(limit);
        functions
    }

    /// Reset profiling data
    pub fn reset(&mut self) {
        self.execution_counts.clear();
        self.total_executions = 0;
    }

    /// Get profiling statistics
    pub fn stats(&self) -> ProfilerStats {
        let hot_functions = self
            .execution_counts
            .iter()
            .filter(|(_, &count)| count >= self.hot_threshold)
            .count();

        ProfilerStats {
            total_functions: self.execution_counts.len(),
            hot_functions,
            total_executions: self.total_executions,
            hot_threshold: self.hot_threshold,
        }
    }
}

impl Default for HotspotProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about profiling activity
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    pub total_functions: usize,
    pub hot_functions: usize,
    pub total_executions: u64,
    pub hot_threshold: u32,
}
