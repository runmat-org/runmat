//! Compact consumption roots shared by VM calls, stores, and frame retirement.
use runmat_accelerate_api::GpuHandleIdentity;
use runmat_gc::RootId;
use runmat_value::Value;
use std::collections::HashSet;

mod arguments;
mod frame;
mod visibility;
#[cfg(test)]
mod tests;
pub(super) use frame::FrameValues;
use visibility::{allocations, release, retain_value, Allocation};

/// Declared before a suspended dispatch future; therefore that future is
/// destroyed before this guard scans the surviving frame and published roots.
struct Registration(RootId);
impl Drop for Registration {
    fn drop(&mut self) {
        let _ = runmat_gc::gc_unregister_root(self.0);
    }
}

/// Public `interpret_with_vars` borrows the caller's inputs. Its error/drop
/// path leaves them untouched; only successful workspace publication retires
/// inputs that the replacement no longer owns.
pub(super) struct BorrowedInputs(ConsumedValues);
impl BorrowedInputs {
    pub fn new(values: &[Value]) -> Result<Self, String> {
        let allocations = allocations(values)?;
        let root = arguments::ArgumentRoot::new(
            values,
            allocations
                .iter()
                .map(|allocation| allocation.handle.clone()),
        );
        let root = if root.is_empty() {
            None
        } else {
            Some(Registration(
                runmat_gc::gc_register_value_root(Box::new(root))
                    .map_err(|error| error.to_string())?,
            ))
        };
        Ok(Self(ConsumedValues {
            allocations: Some(allocations),
            root,
            keep: Some(HashSet::new()),
        }))
    }

    pub fn complete(mut self, outputs: &[Value]) {
        self.0.protect(outputs);
        if let (Some(values), Some(keep)) = (self.0.allocations.take(), &self.0.keep) {
            // Publication is complete. The old compact root is no longer a
            // live caller owner; all other registered owners still participate.
            self.0.root.take();
            release(values, &[], keep);
        }
    }
}
impl Drop for BorrowedInputs {
    fn drop(&mut self) {
        // Abandonment does not consume the caller-owned input workspace.
        self.0.allocations = None;
    }
}

pub(crate) struct ConsumedValues {
    allocations: Option<Vec<Allocation>>,
    root: Option<Registration>,
    keep: Option<HashSet<GpuHandleIdentity>>,
}
impl ConsumedValues {
    pub fn new(values: &[Value]) -> Self {
        match allocations(values) {
            Ok(allocations) => {
                // Suspended callees own the moved arguments. Publish that
                // lifetime through the existing root owner so another nested
                // or interleaved VM frame cannot retire the same alias.
                let root = arguments::ArgumentRoot::new(
                    values,
                    allocations
                        .iter()
                        .map(|allocation| allocation.handle.clone()),
                );
                // Managed arguments may gain resident children during a
                // suspended callback even when no resident child exists yet.
                // Only truly resource-free arguments can skip registration.
                if root.is_empty() {
                    return Self {
                        allocations: None,
                        root: None,
                        keep: Some(HashSet::new()),
                    };
                }
                let root = runmat_gc::gc_register_value_root(Box::new(root));
                match root {
                    Ok(root) => Self {
                        allocations: Some(allocations),
                        root: Some(Registration(root)),
                        keep: Some(HashSet::new()),
                    },
                    Err(reason) => {
                        log::warn!("VM consumed-value registration unavailable: {reason}");
                        Self {
                            allocations: None,
                            root: None,
                            keep: Some(HashSet::new()),
                        }
                    }
                }
            }
            Err(reason) => {
                log::warn!("VM consumed-value ownership unavailable: {reason}");
                Self {
                    allocations: None,
                    root: None,
                    keep: Some(HashSet::new()),
                }
            }
        }
    }
    pub fn protect(&mut self, values: &[Value]) {
        if self.allocations.is_none() {
            return;
        }
        for value in values {
            if let Some(keep) = &mut self.keep {
                if retain_value(value, keep).is_none() {
                    self.keep = None;
                    log::warn!("VM protected output aliases are unavailable");
                    break;
                }
            }
        }
    }
    pub fn finish(mut self, outputs: &[Value]) {
        self.protect(outputs);
    }
}
impl Drop for ConsumedValues {
    fn drop(&mut self) {
        if let (Some(values), Some(keep)) = (self.allocations.take(), &self.keep) {
            let excluded = self.root.iter().map(|root| root.0).collect::<Vec<_>>();
            release(values, &excluded, keep);
        }
    }
}
