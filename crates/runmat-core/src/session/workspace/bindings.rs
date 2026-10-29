//! Compiler slot metadata reconciled with the canonical workspace owner.

use super::binding_control::{BindingLease, BindingPhase};
use super::global_key::matches_session_global_target;
use super::*;

fn binding_control_error(
    error: runmat_runtime::context::host_allocation::HostAllocationError,
) -> RuntimeError {
    runmat_runtime::build_runtime_error(format!("workspace binding authority failed: {error}"))
        .with_identifier("RunMat:WorkspaceBindingSlots")
        .with_source(error)
        .build()
}

fn binding_borrow_error() -> RuntimeError {
    runmat_runtime::build_runtime_error("workspace binding mutation is reentrant")
        .with_identifier("RunMat:WorkspaceBindingSlots")
        .build()
}

impl RunMatSession {
    pub(super) fn interactive_workspace_binding_key(
        &self,
        name: &str,
    ) -> crate::abi::WorkspaceBindingKey {
        crate::abi::WorkspaceBindingKey::Interactive {
            session: self.abi_workspace_handle.0,
            name: runmat_hir::BindingName(name.to_string()),
        }
    }

    /// Refresh compiler/VM slot metadata from a completed invocation without
    /// treating the register file as a second base-workspace authority.
    pub(crate) fn refresh_workspace_binding_slots<'a>(
        &mut self,
        completed: impl IntoIterator<Item = (&'a str, usize)>,
        global_target: impl Fn(
            &str,
            usize,
        ) -> std::result::Result<
            Option<runmat_runtime::workspace::GlobalBindingTarget>,
            RuntimeError,
        >,
    ) -> std::result::Result<(), RuntimeError> {
        let lease = self
            .workspace_bindings
            .reserve(BindingPhase::SlotRefresh, Some(&self.runtime_context))
            .map_err(binding_control_error)?;
        lease
            .checkpoint(Some(&self.runtime_context))
            .map_err(binding_control_error)?;
        for (name, slot) in completed {
            // Every language write has already passed the canonical
            // publication observer. Imported external names are reconciled
            // at request ingress; a completed register file is
            // never authority to allocate a new binding after the effect.
            let global = global_target(name, slot)?;
            self.refresh_existing_workspace_binding(&lease, name, slot, global.as_ref())?;
        }
        Ok(())
    }

    /// Slot refresh does not change an existing non-global origin. Keep the
    /// actual key in place rather than cloning its names and definition path
    /// into a replacement entry after every invocation.
    fn refresh_existing_workspace_binding(
        &self,
        lease: &BindingLease,
        name: &str,
        slot: usize,
        global: Option<&runmat_runtime::workspace::GlobalBindingTarget>,
    ) -> std::result::Result<bool, RuntimeError> {
        let mut bindings = lease.try_borrow_mut().map_err(|_| binding_borrow_error())?;
        let Some(binding) = bindings.get_mut(name) else {
            return Ok(false);
        };
        let keeps_origin = match global {
            Some(target) => {
                matches_session_global_target(&binding.key, self.abi_workspace_handle.0, target)
            }
            None => !binding.key.is_global(),
        };
        if keeps_origin {
            binding.slot = slot;
            lease.mark_changed();
        }
        Ok(keeps_origin)
    }
}
