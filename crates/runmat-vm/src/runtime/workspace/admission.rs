use super::{
    BaseTargetAuthority, PendingWorkspace, WorkspaceAssignedReport, WorkspaceFramePolicy,
    WorkspaceValueSnapshot,
};
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
use std::collections::{HashMap, HashSet};

/// Exact workspace input consumed by one interpreter invocation.
///
/// This is moved into the VM service frame before the future can yield, so
/// concurrently pending invocations sharing a RuntimeContext cannot consume
/// each other's workspace state.
pub struct PendingWorkspaceAdmission {
    state: PendingWorkspace,
}

impl PendingWorkspaceAdmission {
    pub(crate) fn into_state(self) -> PendingWorkspace {
        self.state
    }

    fn new(
        names: HashMap<String, usize>,
        assigned: HashSet<String>,
        policy: WorkspaceFramePolicy,
        base_target: BaseTargetAuthority,
        global_aliases: FrameGlobalBindings,
    ) -> Self {
        Self {
            state: PendingWorkspace {
                input: super::PendingWorkspaceInput::Legacy {
                    names,
                    assigned,
                    global_aliases,
                },
                policy,
                base_target,
            },
        }
    }

    pub(crate) fn admitted(metadata: super::admitted_target::AdmittedWorkspaceMetadata) -> Self {
        Self {
            state: PendingWorkspace {
                input: super::PendingWorkspaceInput::Admitted(metadata),
                policy: WorkspaceFramePolicy::Transactional,
                base_target: BaseTargetAuthority::CanonicalService,
            },
        }
    }
}

pub fn workspace_admission(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
) -> PendingWorkspaceAdmission {
    workspace_admission_with_global_bindings(names, assigned, FrameGlobalBindings::default())
}

pub fn workspace_admission_with_global_bindings(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    global_aliases: FrameGlobalBindings,
) -> PendingWorkspaceAdmission {
    PendingWorkspaceAdmission::new(
        names,
        assigned,
        WorkspaceFramePolicy::TopLevel,
        BaseTargetAuthority::Frame,
        global_aliases,
    )
}

/// An isolated source-evaluation projection publishes only to its detached
/// projection; an explicit `base` selector instead requires the session owner.
pub(crate) fn detached_workspace_admission_with_global_bindings(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    global_aliases: FrameGlobalBindings,
) -> PendingWorkspaceAdmission {
    PendingWorkspaceAdmission::new(
        names,
        assigned,
        WorkspaceFramePolicy::TopLevel,
        BaseTargetAuthority::CanonicalService,
        global_aliases,
    )
}

/// Numeric-text evaluation owns a detached *transactional* frame. Keep its
/// completion and no-publication policy while selecting the canonical base
/// owner only for an explicit Base operation.
pub(crate) fn detached_transactional_workspace_admission_with_global_bindings(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    global_aliases: FrameGlobalBindings,
) -> PendingWorkspaceAdmission {
    PendingWorkspaceAdmission::new(
        names,
        assigned,
        WorkspaceFramePolicy::Transactional,
        BaseTargetAuthority::CanonicalService,
        global_aliases,
    )
}

pub(crate) fn transactional_workspace_admission_with_global_bindings(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    global_aliases: FrameGlobalBindings,
) -> PendingWorkspaceAdmission {
    PendingWorkspaceAdmission::new(
        names,
        assigned,
        WorkspaceFramePolicy::Transactional,
        BaseTargetAuthority::Frame,
        global_aliases,
    )
}

#[derive(Debug)]
pub struct WorkspaceExecutionCompletion {
    pub state: WorkspaceValueSnapshot,
    pub assigned: WorkspaceAssignedReport,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_text_keeps_transactional_policy_with_explicit_base_authority() {
        let detached = detached_transactional_workspace_admission_with_global_bindings(
            HashMap::new(),
            HashSet::new(),
            FrameGlobalBindings::default(),
        )
        .into_state();
        assert_eq!(detached.policy, WorkspaceFramePolicy::Transactional);
        assert_eq!(detached.base_target, BaseTargetAuthority::CanonicalService);

        let ordinary = transactional_workspace_admission_with_global_bindings(
            HashMap::new(),
            HashSet::new(),
            FrameGlobalBindings::default(),
        )
        .into_state();
        assert_eq!(ordinary.policy, WorkspaceFramePolicy::Transactional);
        assert_eq!(ordinary.base_target, BaseTargetAuthority::Frame);
    }
}
