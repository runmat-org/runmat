//! GC root scanning and management
//!
//! Handles the identification and scanning of GC roots - objects that
//! should not be collected because they are reachable from the program's
//! execution context (stacks, global variables, etc.).

use crate::Value;
use crate::{GcError, GcHandle, Result};
use runmat_gc_api::{Trace, Tracer};
use runmat_time::Instant;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Recursively collect immediate GC roots from an owned Value tree.
///
/// This helper deliberately does not dereference collected `GcHandle`s. Root
/// scanning should discover handles reachable from ordinary Rust-owned values;
/// the collector is responsible for tracing each discovered GC object.
pub(crate) fn collect_value_roots(value: &Value, roots: &mut Vec<GcHandle>) {
    struct RootCollector<'a> {
        roots: &'a mut Vec<GcHandle>,
    }

    impl Tracer for RootCollector<'_> {
        fn mark(&mut self, handle: GcHandle) {
            self.roots.push(handle);
        }
    }

    value.trace(&mut RootCollector { roots });
}

/// Unique identifier for a GC root
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RootId(pub usize);

/// Trait for objects that can serve as GC roots
pub trait GcRoot {
    /// Scan this root and return all reachable GC handles.
    fn scan(&self) -> Vec<GcHandle>;

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

impl GcRoot for StackRoot {
    fn scan(&self) -> Vec<GcHandle> {
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

impl GcRoot for VariableArrayRoot {
    fn scan(&self) -> Vec<GcHandle> {
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
    fn scan(&self) -> Vec<GcHandle> {
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
    pub fn scan_roots(&self) -> Result<Vec<GcHandle>> {
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
    use runmat_builtins::{Closure, HandleRef, Listener, ObjectInstance, StructValue};

    fn ptr_addr(ptr: GcHandle) -> usize {
        ptr.addr()
    }

    #[test]
    fn test_global_root() {
        let values = vec![Value::Num(42.0), Value::String("test".to_string())];

        let root = GlobalRoot::new(values, "test global".to_string());
        assert_eq!(root.description(), "test global");
        assert!(root.is_active());

        let scanned = root.scan();
        assert_eq!(scanned.len(), 0);
    }

    #[test]
    fn global_root_scans_direct_handle_and_listener_targets() {
        crate::gc_reset_for_test().expect("reset gc");
        let target = crate::gc_allocate(Value::Object(ObjectInstance::new("widget".to_string())))
            .expect("target allocation");
        let callback = crate::gc_allocate(Value::FunctionHandle("onEvent".to_string()))
            .expect("callback allocation");

        let values = vec![
            Value::HandleObject(HandleRef {
                class_name: "widget".to_string(),
                target: target.clone(),
                valid: true,
            }),
            Value::Listener(Listener {
                id: 1,
                target: target.clone(),
                event_name: "Changed".to_string(),
                callback: callback.clone(),
                enabled: true,
                valid: true,
            }),
        ];

        let root = GlobalRoot::new(values, "handles".to_string());
        let scanned = root.scan();
        let scanned_addrs: Vec<usize> = scanned.into_iter().map(ptr_addr).collect();

        assert!(scanned_addrs.contains(&ptr_addr(target)));
        assert!(scanned_addrs.contains(&ptr_addr(callback)));
    }

    #[test]
    fn global_root_scans_owned_nested_value_containers() {
        crate::gc_reset_for_test().expect("reset gc");
        let handle_target =
            crate::gc_allocate(Value::Object(ObjectInstance::new("nested".to_string())))
                .expect("target allocation");
        let callback = crate::gc_allocate(Value::FunctionHandle("cb".to_string()))
            .expect("callback allocation");

        let handle = Value::HandleObject(HandleRef {
            class_name: "nested".to_string(),
            target: handle_target.clone(),
            valid: true,
        });
        let listener = Value::Listener(Listener {
            id: 7,
            target: handle_target.clone(),
            event_name: "Changed".to_string(),
            callback: callback.clone(),
            enabled: true,
            valid: true,
        });

        let mut object = ObjectInstance::new("owner".to_string());
        object.properties.insert("h".to_string(), handle.clone());

        let mut struct_value = StructValue::new();
        struct_value.fields.insert(
            "closure".to_string(),
            Value::Closure(Closure {
                function_name: "f".to_string(),
                bound_function: None,
                captures: vec![listener],
            }),
        );
        struct_value
            .fields
            .insert("object".to_string(), Value::Object(object));

        let root = GlobalRoot::new(vec![Value::Struct(struct_value)], "nested".to_string());
        let scanned = root.scan();
        let scanned_addrs: Vec<usize> = scanned.into_iter().map(ptr_addr).collect();

        assert!(scanned_addrs.contains(&ptr_addr(handle_target)));
        assert!(scanned_addrs.contains(&ptr_addr(callback)));
    }

    #[test]
    fn test_root_scanner() {
        let scanner = RootScanner::new();

        let root = Box::new(GlobalRoot::new(vec![Value::Num(1.0)], "test".to_string()));

        let root_id = scanner.register_root(root).expect("should register");

        let roots = scanner.scan_roots().expect("should scan");
        assert_eq!(roots.len(), 0); // No GC handles yet

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
        assert_eq!(scanned.len(), 0);
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
