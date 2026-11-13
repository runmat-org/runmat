//! Active global aliases of one execution frame.
//!
//! The VM and native executor share this authority with their dynamic-workspace
//! adapter. A detached snapshot is used only when transferring a frame across
//! execution engines; it must not become a second mutable alias store.

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameGlobalBinding {
    Named(String),
    LegacySlot(usize),
}

/// Exact session destination of one live workspace-global binding. The
/// workspace spelling is a separate key; a named target may differ from it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlobalBindingTarget {
    Named(crate::workspace::WorkspaceName),
    LegacySlot(usize),
}

impl GlobalBindingTarget {
    pub(crate) fn additional_retained_bytes(
        &self,
    ) -> Result<u64, crate::context::host_allocation::HostAllocationError> {
        use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};
        match self {
            Self::Named(name) if !name.has_retained_claim() => {
                let bytes = u64::try_from(crate::workspace::WorkspaceName::requested_bytes(
                    name.as_str(),
                )?)
                .map_err(|_| HostAllocationError::Overflow)?;
                HostAllocationPlan::retained_exact_output_bytes(bytes, 1)
            }
            _ => Ok(0),
        }
    }

    pub(crate) fn prepare_retained(
        &self,
        permit: &dyn runmat_value::ExactHostAllocationPermit,
    ) -> Result<Self, runmat_value::ExactHostNumericAllocationError> {
        match self {
            Self::Named(name) if !name.has_retained_claim() => Ok(Self::Named(
                crate::workspace::WorkspaceName::prepare(name.as_str(), permit)?,
            )),
            _ => Ok(self.clone()),
        }
    }
}

mod owner;
pub use owner::{
    FrameGlobalBindings, FrameGlobalBindingsRetirement, PreparedAliasBinding,
    PreparedAliasReplacement, PreparedAliasReplacementRetirement,
};

#[cfg(test)]
mod tests {
    use super::FrameGlobalBindings;
    use crate::{context::RuntimeContext, execution::RuntimeExecutionService};
    use std::rc::Rc;

    #[test]
    fn clones_share_frame_authority_but_snapshots_are_detached() {
        let owner = FrameGlobalBindings::default();
        let adapter = owner.clone();
        owner.declare_named(2, "alpha".into());
        owner.declare_legacy_slot(3);
        assert!(adapter.has_binding_for_slot(2));
        assert!(adapter.has_binding_for_slot(3));
        assert!(!adapter.has_binding_for_slot(4));
        assert!(adapter.is_bound(2, "alpha"));
        let snapshot = owner.snapshot();
        adapter.unbind_slot(2);
        assert!(!owner.has_binding_for_slot(2));
        assert_eq!(owner.name_for_slot(2), None);
        assert_eq!(
            snapshot.get(&2),
            Some(&super::FrameGlobalBinding::Named("alpha".into()))
        );
        assert_eq!(
            snapshot.get(&3),
            Some(&super::FrameGlobalBinding::LegacySlot(3))
        );
        let resumed = FrameGlobalBindings::from_snapshot(snapshot);
        resumed.declare_named(2, "beta".into());
        assert_eq!(resumed.name_for_slot(2).as_deref(), Some("beta"));
        assert_eq!(owner.name_for_slot(2), None);
        assert_eq!(
            resumed.binding_for_slot(3),
            Some(super::FrameGlobalBinding::LegacySlot(3))
        );
    }

    #[test]
    fn admitted_alias_retains_disjoint_legacy_target_slot() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let aliases = FrameGlobalBindings::default();
        let prepared = aliases
            .prepare_target(&runtime, 17, &super::GlobalBindingTarget::LegacySlot(4))
            .unwrap();
        assert_eq!(prepared.target(), super::GlobalBindingTarget::LegacySlot(4));
        prepared.checkpoint().unwrap();
        prepared.commit();
        assert_eq!(
            aliases.binding_handle_for_slot(17),
            Some(super::GlobalBindingTarget::LegacySlot(4))
        );
        assert_eq!(
            aliases.snapshot().get(&17),
            Some(&super::FrameGlobalBinding::LegacySlot(4))
        );
    }

    #[test]
    fn prepared_alias_move_preserves_parent_handle_and_fences_reentry() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let parent = FrameGlobalBindings::default();
        let observer = parent.clone();
        parent.declare_named(1, "old".into());
        let child = FrameGlobalBindings::default();
        let named = child.prepare_named(&runtime, 2, "new").unwrap();
        assert!(child.prepare_legacy_slot(&runtime, 3).is_err());
        named.checkpoint().unwrap();
        named.commit();
        let replacement = parent.prepare_replace_with_owned(child.clone()).unwrap();
        assert!(parent.prepare_named(&runtime, 4, "blocked").is_err());
        replacement.checkpoint().unwrap();
        replacement.commit();
        assert_eq!(observer.name_for_slot(2).as_deref(), Some("new"));
        assert!(!observer.has_binding_for_slot(1));
        assert!(!child.has_binding_for_slot(2));
    }
}
