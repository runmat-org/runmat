//! GC root scanning and management
//!
//! Handles the identification and scanning of GC roots - objects that
//! should not be collected because they are reachable from the program's
//! execution context (stacks, global variables, etc.).

use crate::Value;
use crate::{GcError, GcPtr, Result};
use runmat_time::Instant;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Recursively collect GC roots from a Value.
///
/// This is a shared helper used by all root types to traverse nested
/// values (cells and structs) and collect GC pointers.
fn collect_value_roots(value: &Value, roots: &mut Vec<GcPtr<Value>>) {
    match value {
        Value::Cell(cells) => {
            for cell_value in &cells.data {
                roots.push(cell_value.clone());
                let inner = unsafe { &*cell_value.as_raw() };
                collect_value_roots(inner, roots);
            }
        }
        Value::Struct(struct_value) => {
            for field_value in struct_value.fields.values() {
                collect_value_roots(field_value, roots);
            }
        }
        _ => {}
    }
}

/// Unique identifier for a GC root
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RootId(pub usize);

/// Trait for objects that can serve as GC roots
pub trait GcRoot: Send + Sync {
    /// Scan this root and return all reachable GC pointers
    fn scan(&self) -> Vec<GcPtr<Value>>;

    /// Get a human-readable description of this root
    fn description(&self) -> String;

    /// Get the estimated size of objects reachable from this root
    fn estimated_size(&self) -> usize {
        0 // Default implementation
    }

    /// Check if this root is still active
    fn is_active(&self) -> bool {
        true // Most roots are always active
    }
}

/// A root representing an interpreter's value stack
pub struct StackRoot {
    /// Reference to the stack (non-owning)
    stack_ptr: *const Vec<Value>,
    description: String,
}

impl StackRoot {
    /// Create a new stack root
    ///
    /// # Safety
    ///
    /// The stack pointer must remain valid for the lifetime of this root
    pub unsafe fn new(stack: *const Vec<Value>, description: String) -> Self {
        Self {
            stack_ptr: stack,
            description,
        }
    }
}

// Safety: StackRoot is used in a single-threaded context where the pointer
// remains valid and is not shared across threads unsafely
unsafe impl Send for StackRoot {}
unsafe impl Sync for StackRoot {}

impl GcRoot for StackRoot {
    fn scan(&self) -> Vec<GcPtr<Value>> {
        unsafe {
            if self.stack_ptr.is_null() {
                return Vec::new();
            }

            let stack = &*self.stack_ptr;
            let mut roots = Vec::new();

            for value in stack {
                collect_value_roots(value, &mut roots);
            }

            roots
        }
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn estimated_size(&self) -> usize {
        unsafe {
            if self.stack_ptr.is_null() {
                return 0;
            }

            let stack = &*self.stack_ptr;
            stack.len() * std::mem::size_of::<Value>()
        }
    }

    fn is_active(&self) -> bool {
        !self.stack_ptr.is_null()
    }
}

/// A root representing an array of variables
pub struct VariableArrayRoot {
    /// Reference to the variable array (non-owning)  
    vars_ptr: *const Vec<Value>,
    description: String,
}

impl VariableArrayRoot {
    /// Create a new variable array root
    ///
    /// # Safety
    ///
    /// The variables pointer must remain valid for the lifetime of this root
    pub unsafe fn new(vars: *const Vec<Value>, description: String) -> Self {
        Self {
            vars_ptr: vars,
            description,
        }
    }
}

// Safety: VariableArrayRoot is used in a single-threaded context where the pointer
// remains valid and is not shared across threads unsafely
unsafe impl Send for VariableArrayRoot {}
unsafe impl Sync for VariableArrayRoot {}

impl GcRoot for VariableArrayRoot {
    fn scan(&self) -> Vec<GcPtr<Value>> {
        unsafe {
            if self.vars_ptr.is_null() {
                return Vec::new();
            }

            let vars = &*self.vars_ptr;
            let mut roots = Vec::new();

            for value in vars {
                collect_value_roots(value, &mut roots);
            }

            roots
        }
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn estimated_size(&self) -> usize {
        unsafe {
            if self.vars_ptr.is_null() {
                return 0;
            }

            let vars = &*self.vars_ptr;
            vars.len() * std::mem::size_of::<Value>()
        }
    }

    fn is_active(&self) -> bool {
        !self.vars_ptr.is_null()
    }
}

/// A root for global/static values
pub struct GlobalRoot {
    /// Owned global values
    values: Vec<Value>,
    description: String,
}

impl GlobalRoot {
    pub fn new(values: Vec<Value>, description: String) -> Self {
        Self {
            values,
            description,
        }
    }

    pub fn add_value(&mut self, value: Value) {
        self.values.push(value);
    }

    pub fn remove_value(&mut self, index: usize) -> Option<Value> {
        if index < self.values.len() {
            Some(self.values.remove(index))
        } else {
            None
        }
    }
}

impl GcRoot for GlobalRoot {
    fn scan(&self) -> Vec<GcPtr<Value>> {
        let mut roots = Vec::new();
        for value in &self.values {
            collect_value_roots(value, &mut roots);
        }
        roots
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn estimated_size(&self) -> usize {
        self.values.len() * std::mem::size_of::<Value>()
    }
}

/// Manages all GC roots in the system
pub struct RootScanner {
    /// Registered roots
    roots: parking_lot::RwLock<HashMap<RootId, Box<dyn GcRoot>>>,

    /// Next root ID to assign
    next_id: AtomicUsize,

    /// Statistics
    scans_performed: AtomicUsize,
    total_roots_found: AtomicUsize,
}

impl RootScanner {
    pub fn new() -> Self {
        Self {
            roots: parking_lot::RwLock::new(HashMap::new()),
            next_id: AtomicUsize::new(1),
            scans_performed: AtomicUsize::new(0),
            total_roots_found: AtomicUsize::new(0),
        }
    }

    /// Register a new GC root
    pub fn register_root(&self, root: Box<dyn GcRoot>) -> Result<RootId> {
        let id = RootId(self.next_id.fetch_add(1, Ordering::Relaxed));

        log::debug!("Registering GC root {}: {}", id.0, root.description());

        let mut roots = self.roots.write();
        roots.insert(id, root);

        Ok(id)
    }

    /// Unregister a GC root
    pub fn unregister_root(&self, root_id: RootId) -> Result<()> {
        log::debug!("Unregistering GC root {}", root_id.0);

        let mut roots = self.roots.write();
        match roots.remove(&root_id) {
            Some(_) => Ok(()),
            None => Err(GcError::RootRegistrationFailed(format!(
                "Root {} not found",
                root_id.0
            ))),
        }
    }

    /// Scan all roots and return reachable objects
    pub fn scan_roots(&self) -> Result<Vec<GcPtr<Value>>> {
        log::trace!("Starting root scan");
        let scan_start = Instant::now();

        let roots = self.roots.read();
        let mut all_roots = Vec::new();
        let mut inactive_roots = Vec::new();

        for (&root_id, root) in roots.iter() {
            if !root.is_active() {
                inactive_roots.push(root_id);
                continue;
            }

            log::trace!("Scanning root {}: {}", root_id.0, root.description());
            let root_objects = root.scan();
            log::trace!(
                "Found {} objects from root {}",
                root_objects.len(),
                root_id.0
            );

            all_roots.extend(root_objects);
        }

        drop(roots);

        // Clean up inactive roots
        if !inactive_roots.is_empty() {
            let mut roots = self.roots.write();
            for root_id in inactive_roots {
                log::debug!("Removing inactive root {}", root_id.0);
                roots.remove(&root_id);
            }
        }

        // Update statistics
        self.scans_performed.fetch_add(1, Ordering::Relaxed);
        self.total_roots_found
            .fetch_add(all_roots.len(), Ordering::Relaxed);

        let scan_duration = scan_start.elapsed();
        log::debug!(
            "Root scan completed: {} roots found in {:?}",
            all_roots.len(),
            scan_duration
        );

        Ok(all_roots)
    }

    /// Get information about all registered roots
    pub fn root_info(&self) -> Vec<RootInfo> {
        let roots = self.roots.read();
        roots
            .iter()
            .map(|(&id, root)| RootInfo {
                id,
                description: root.description(),
                estimated_size: root.estimated_size(),
                is_active: root.is_active(),
            })
            .collect()
    }

    /// Get scanner statistics
    pub fn stats(&self) -> RootScannerStats {
        let roots = self.roots.read();
        RootScannerStats {
            registered_roots: roots.len(),
            scans_performed: self.scans_performed.load(Ordering::Relaxed),
            total_roots_found: self.total_roots_found.load(Ordering::Relaxed),
            average_roots_per_scan: if self.scans_performed.load(Ordering::Relaxed) > 0 {
                self.total_roots_found.load(Ordering::Relaxed) as f64
                    / self.scans_performed.load(Ordering::Relaxed) as f64
            } else {
                0.0
            },
        }
    }

    /// Remove all inactive roots
    pub fn cleanup_inactive_roots(&self) -> usize {
        let mut roots = self.roots.write();
        let initial_count = roots.len();

        roots.retain(|_, root| root.is_active());

        let removed_count = initial_count - roots.len();
        if removed_count > 0 {
            log::debug!("Cleaned up {removed_count} inactive roots");
        }

        removed_count
    }
}

impl Default for RootScanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a registered root
#[derive(Debug, Clone)]
pub struct RootInfo {
    pub id: RootId,
    pub description: String,
    pub estimated_size: usize,
    pub is_active: bool,
}

/// Statistics for the root scanner
#[derive(Debug, Clone)]
pub struct RootScannerStats {
    pub registered_roots: usize,
    pub scans_performed: usize,
    pub total_roots_found: usize,
    pub average_roots_per_scan: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_root() {
        let values = vec![Value::Num(42.0), Value::String("test".to_string())];

        let root = GlobalRoot::new(values, "test global".to_string());
        assert_eq!(root.description(), "test global");
        assert!(root.is_active());

        let scanned = root.scan();
        // Currently returns empty since we don't have GC-allocated Values yet
        assert_eq!(scanned.len(), 0);
    }

    #[test]
    fn test_root_scanner() {
        let scanner = RootScanner::new();

        let root = Box::new(GlobalRoot::new(vec![Value::Num(1.0)], "test".to_string()));

        let root_id = scanner.register_root(root).expect("should register");

        let roots = scanner.scan_roots().expect("should scan");
        assert_eq!(roots.len(), 0); // No GC pointers yet

        let info = scanner.root_info();
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].description, "test");

        scanner.unregister_root(root_id).expect("should unregister");

        let info = scanner.root_info();
        assert_eq!(info.len(), 0);
    }

    #[test]
    fn test_stack_root() {
        let stack = vec![Value::Num(1.0), Value::Bool(true)];

        let root = unsafe { StackRoot::new(&stack as *const _, "test stack".to_string()) };

        assert_eq!(root.description(), "test stack");
        assert!(root.is_active());
        assert!(root.estimated_size() > 0);

        let scanned = root.scan();
        assert_eq!(scanned.len(), 0); // No GC pointers in current implementation
    }

    #[test]
    fn test_variable_array_root() {
        let vars = vec![Value::Num(42.0), Value::String("test".to_string())];

        let root = unsafe { VariableArrayRoot::new(&vars as *const _, "test vars".to_string()) };

        assert_eq!(root.description(), "test vars");
        assert!(root.is_active());
        assert!(root.estimated_size() > 0);
    }

    #[test]
    fn test_root_scanner_stats() {
        let scanner = RootScanner::new();

        let initial_stats = scanner.stats();
        assert_eq!(initial_stats.registered_roots, 0);
        assert_eq!(initial_stats.scans_performed, 0);

        let _root_id = scanner
            .register_root(Box::new(GlobalRoot::new(vec![], "test".to_string())))
            .expect("should register");

        let _roots = scanner.scan_roots().expect("should scan");

        let stats = scanner.stats();
        assert_eq!(stats.registered_roots, 1);
        assert_eq!(stats.scans_performed, 1);
    }
}
