use crate::GcHandle;

/// Unique identifier for a registered GC root source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RootId(pub usize);

/// Source of GC handles that must be treated as roots during collection.
pub trait GcRoot {
    /// Scan this root source and return all reachable GC handles.
    fn scan(&self) -> Vec<GcHandle>;

    /// Get a human-readable description of this root source.
    fn description(&self) -> String;

    /// Get the estimated size of values reachable from this root source.
    fn estimated_size(&self) -> usize {
        0
    }

    /// Check if this root source is still active.
    fn is_active(&self) -> bool {
        true
    }
}

/// Information about a registered root source.
#[derive(Debug, Clone)]
pub struct RootInfo {
    pub id: RootId,
    pub description: String,
    pub estimated_size: usize,
    pub is_active: bool,
}

/// Statistics for root scanning.
#[derive(Debug, Clone)]
pub struct RootScannerStats {
    pub registered_roots: usize,
    pub scans_performed: usize,
    pub total_roots_found: usize,
    pub average_roots_per_scan: f64,
}
