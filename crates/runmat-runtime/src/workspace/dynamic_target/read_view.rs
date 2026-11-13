//! Terminal, immutable portion of an exact dynamic workspace capture.
//!
//! A completed result retains the same logical generation and lexical
//! overrides, but no longer owns a mutation issuer or completion identity.

use runmat_value::Value;

use super::DynamicWorkspaceTargetOverlay;
use crate::workspace::{GlobalBindingTarget, LogicalWorkspaceGeneration, WorkspaceSlotReadLease};

#[derive(Clone, Debug)]
pub struct CapturedWorkspaceReadView {
    generation: LogicalWorkspaceGeneration,
    overlay: DynamicWorkspaceTargetOverlay,
}

impl CapturedWorkspaceReadView {
    pub(super) fn new(
        generation: LogicalWorkspaceGeneration,
        overlay: DynamicWorkspaceTargetOverlay,
    ) -> Self {
        Self {
            generation,
            overlay,
        }
    }

    pub(super) fn with_overlay(self, overlay: DynamicWorkspaceTargetOverlay) -> Self {
        Self {
            generation: self.generation,
            overlay,
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        if let Some(entry) = self.overlay.get(name) {
            return entry.value().map(WorkspaceSlotReadLease::value);
        }
        self.generation.get(name)
    }

    pub fn read_lease(&self, name: &str) -> Option<WorkspaceSlotReadLease> {
        if let Some(entry) = self.overlay.get(name) {
            return entry.value().cloned();
        }
        self.generation
            .read_lease(name)
            .map(WorkspaceSlotReadLease::from)
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        if self.overlay.get(name).is_some() {
            return None;
        }
        self.generation.global_target(name)
    }

    pub fn global_bindings(&self) -> impl Iterator<Item = (&str, &GlobalBindingTarget)> {
        self.generation
            .global_bindings()
            .filter(|(name, _)| self.overlay.get(name).is_none())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        let mut base = self.generation.iter().peekable();
        let mut overlay = self.overlay.iter().peekable();
        std::iter::from_fn(move || loop {
            let base_name = base.peek().map(|(name, _)| *name);
            let overlay_name = overlay.peek().map(|entry| entry.name());
            match (base_name, overlay_name) {
                (Some(base_name), Some(overlay_name)) if base_name == overlay_name => {
                    base.next();
                    if let Some(entry) = overlay.next() {
                        if let Some(value) = entry.value() {
                            return Some((entry.name(), value.value()));
                        }
                    }
                }
                (Some(base_name), Some(overlay_name)) if overlay_name < base_name => {
                    if let Some(entry) = overlay.next() {
                        if let Some(value) = entry.value() {
                            return Some((entry.name(), value.value()));
                        }
                    }
                }
                (Some(_), _) => return base.next(),
                (None, Some(_)) => {
                    if let Some(entry) = overlay.next() {
                        if let Some(value) = entry.value() {
                            return Some((entry.name(), value.value()));
                        }
                    }
                }
                (None, None) => return None,
            }
        })
    }
}
