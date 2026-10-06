use super::{
    PendingWorkspace, WorkspaceAssignedReport, WorkspaceFramePolicy, WorkspaceValueSnapshot,
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
        global_aliases: FrameGlobalBindings,
    ) -> Self {
        Self {
            state: PendingWorkspace {
                names,
                assigned,
                policy,
                global_aliases,
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
        global_aliases,
    )
}

#[derive(Debug, Clone)]
pub struct WorkspaceExecutionCompletion {
    pub state: WorkspaceValueSnapshot,
    pub assigned: WorkspaceAssignedReport,
}
