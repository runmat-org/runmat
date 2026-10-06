use super::*;
use std::rc::Weak;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct WorkspaceFrameId(pub(crate) usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceFrameLifecycle {
    Pending,
    Active,
    Closed,
}

pub(crate) struct PendingServiceFrame {
    pub(in crate::runtime) id: WorkspaceFrameId,
    pub(in crate::runtime) lifecycle: Rc<Cell<WorkspaceFrameLifecycle>>,
    pub(in crate::runtime) delegate:
        Rc<RefCell<Option<Rc<dyn runmat_runtime::context::RuntimeWorkspaceService>>>>,
    pub(in crate::runtime) service:
        Rc<RefCell<Option<Weak<dyn runmat_runtime::context::RuntimeWorkspaceService>>>>,
    pub(in crate::runtime) parent: Rc<Cell<Option<WorkspaceFrameId>>>,
    pub(in crate::runtime) base: Rc<Cell<Option<WorkspaceFrameId>>>,
    pub(in crate::runtime) admission: Rc<RefCell<Option<PendingWorkspace>>>,
    pub(in crate::runtime) session_variables:
        Rc<RefCell<Option<runmat_runtime::workspace::session::SessionVariableAuthority>>>,
}

pub(crate) fn reserve_workspace_service_frame(
    runtime: runmat_runtime::context::RuntimeContextLocalId,
    admission: Option<PendingWorkspace>,
) -> PendingServiceFrame {
    let id = NEXT_WORKSPACE_FRAME_ID.with(|next| {
        let id = next.get();
        next.set(
            id.checked_add(1)
                .expect("workspace frame identity exhausted"),
        );
        WorkspaceFrameId(id)
    });
    let lifecycle = Rc::new(Cell::new(WorkspaceFrameLifecycle::Pending));
    let delegate = Rc::new(RefCell::new(None));
    let service = Rc::new(RefCell::new(None));
    let parent = Rc::new(Cell::new(None));
    let base = Rc::new(Cell::new(None));
    let admission = Rc::new(RefCell::new(admission));
    let session_variables = Rc::new(RefCell::new(None));
    PENDING_SERVICE_FRAMES.with(|frames| {
        frames
            .borrow_mut()
            .entry(WorkspaceOwner::Runtime(runtime))
            .or_default()
            .push(PendingServiceFrame {
                id,
                lifecycle: lifecycle.clone(),
                delegate: delegate.clone(),
                service: service.clone(),
                parent: parent.clone(),
                base: base.clone(),
                admission: admission.clone(),
                session_variables: session_variables.clone(),
            });
    });
    PendingServiceFrame {
        id,
        lifecycle,
        delegate,
        service,
        parent,
        base,
        admission,
        session_variables,
    }
}

pub(crate) fn release_pending_workspace_service_frame(
    runtime: runmat_runtime::context::RuntimeContextLocalId,
    id: WorkspaceFrameId,
) {
    release_pending_workspace_service_frame_for_owner(WorkspaceOwner::Runtime(runtime), id);
}

pub(super) fn take_service_frame(owner: WorkspaceOwner) -> PendingServiceFrame {
    let active = active_frame(owner);
    PENDING_SERVICE_FRAMES
        .with(|frames| {
            let mut frames = frames.borrow_mut();
            let pending = frames.get_mut(&owner).and_then(|pending| {
                let active = active?;
                let index = pending.iter().position(|frame| frame.id == active)?;
                Some(pending.remove(index))
            });
            if frames.get(&owner).is_some_and(Vec::is_empty) {
                frames.remove(&owner);
            }
            pending
        })
        .unwrap_or_else(|| {
            let (id, lifecycle) = match owner {
                WorkspaceOwner::Runtime(_) => {
                    panic!("VM workspace entered without its exact reserved service frame")
                }
                WorkspaceOwner::Ambient => {
                    let id = NEXT_WORKSPACE_FRAME_ID.with(|next| {
                        let id = next.get();
                        next.set(
                            id.checked_add(1)
                                .expect("workspace frame identity exhausted"),
                        );
                        WorkspaceFrameId(id)
                    });
                    (id, Rc::new(Cell::new(WorkspaceFrameLifecycle::Pending)))
                }
            };
            release_pending_workspace_service_frame_for_owner(owner, id);
            PendingServiceFrame {
                id,
                lifecycle,
                delegate: Rc::new(RefCell::new(None)),
                service: Rc::new(RefCell::new(None)),
                parent: Rc::new(Cell::new(None)),
                base: Rc::new(Cell::new(None)),
                admission: Rc::new(RefCell::new(None)),
                session_variables: Rc::new(RefCell::new(None)),
            }
        })
}

pub(super) fn take_service_frame_admission(owner: WorkspaceOwner) -> Option<PendingWorkspace> {
    let active = active_frame(owner)?;
    PENDING_SERVICE_FRAMES.with(|frames| {
        let frames = frames.borrow();
        let frame = frames
            .get(&owner)?
            .iter()
            .find(|frame| frame.id == active)?;
        let admission = frame.admission.borrow_mut().take();
        admission
    })
}

pub(crate) fn workspace_frame_for_service(
    runtime: runmat_runtime::context::RuntimeContextLocalId,
    service: &Rc<dyn runmat_runtime::context::RuntimeWorkspaceService>,
) -> Option<(WorkspaceFrameId, WorkspaceFrameId)> {
    WORKSPACE_STACKS.with(|stacks| {
        stacks
            .borrow()
            .get(&WorkspaceOwner::Runtime(runtime))?
            .iter()
            .find_map(|frame| {
                let candidate = frame.service.as_ref()?.upgrade()?;
                Rc::ptr_eq(&candidate, service).then_some((frame.id, frame.base))
            })
    })
}

fn release_pending_workspace_service_frame_for_owner(owner: WorkspaceOwner, id: WorkspaceFrameId) {
    PENDING_SERVICE_FRAMES.with(|frames| {
        let mut frames = frames.borrow_mut();
        if let Some(pending) = frames.get_mut(&owner) {
            pending.retain(|frame| frame.id != id);
            if pending.is_empty() {
                frames.remove(&owner);
            }
        }
    });
}

fn with_frame<R>(id: WorkspaceFrameId, f: impl FnOnce(&WorkspaceFrame) -> R) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let stacks = stacks.borrow();
        stacks
            .values()
            .flat_map(|stack| stack.iter())
            .find(|frame| frame.id == id)
            .map(f)
    })
}

fn with_frame_mut<R>(id: WorkspaceFrameId, f: impl FnOnce(&mut WorkspaceFrame) -> R) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.borrow_mut();
        stacks
            .values_mut()
            .flat_map(|stack| stack.iter_mut())
            .find(|frame| frame.id == id)
            .map(f)
    })
}

pub(crate) fn workspace_frame_lookup(id: WorkspaceFrameId, name: &str) -> Option<Value> {
    with_frame(id, |frame| {
        let index = frame.state.names.get(name)?;
        frame
            .state
            .assigned
            .contains(name)
            .then(|| frame.state.values.get(*index).cloned())
            .flatten()
    })
    .flatten()
}

pub(crate) fn workspace_frame_snapshot(id: WorkspaceFrameId) -> Option<Vec<(String, Value)>> {
    with_frame(id, |frame| {
        let mut entries = frame
            .state
            .names
            .iter()
            .filter_map(|(name, index)| {
                frame
                    .state
                    .assigned
                    .contains(name)
                    .then(|| frame.state.values.get(*index).cloned())
                    .flatten()
                    .map(|value| (name.clone(), value))
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        entries
    })
}

pub(crate) fn workspace_frame_publishes(id: WorkspaceFrameId) -> Option<bool> {
    with_frame(id, |frame| frame.policy.publishes_mutations())
}

pub(crate) fn workspace_frame_name_is_global(id: WorkspaceFrameId, name: &str) -> Option<bool> {
    with_frame(id, |frame| {
        frame
            .state
            .names
            .get(name)
            .is_some_and(|slot| frame.global_aliases.binding_for_slot(*slot).is_some())
    })
}

pub(crate) fn workspace_frame_declare_global(
    id: WorkspaceFrameId,
    slot: usize,
    name: String,
) -> Option<()> {
    with_frame_mut(id, |frame| {
        frame.global_aliases.declare_named(slot, name);
    })
}

pub(crate) fn workspace_frame_assign(
    id: WorkspaceFrameId,
    name: &str,
    value: Value,
) -> Result<(), String> {
    with_frame_mut(id, |frame| {
        set_workspace_variable_in_frame_with_reconciliation(frame, name, value, true, false)
    })
    .ok_or_else(|| "workspace frame is closed".to_string())?
}

pub(crate) fn workspace_frame_assign_from_executor(
    id: WorkspaceFrameId,
    name: &str,
    value: Value,
) -> Result<(), String> {
    with_frame_mut(id, |frame| {
        set_workspace_variable_in_frame_with_reconciliation(frame, name, value, false, false)
    })
    .ok_or_else(|| "workspace frame is closed".to_string())?
}

pub(crate) fn workspace_frame_clear(id: WorkspaceFrameId) -> Result<(), String> {
    with_frame_mut(id, clear_workspace_frame).ok_or_else(|| "workspace frame is closed".to_string())
}

pub(crate) fn workspace_frame_remove(id: WorkspaceFrameId, name: &str) -> Result<(), String> {
    with_frame_mut(id, |frame| remove_workspace_name(frame, name))
        .ok_or_else(|| "workspace frame is closed".to_string())
}

pub(crate) fn reconcile_workspace_frame(
    id: WorkspaceFrameId,
    values: Vec<(String, Value)>,
) -> Result<(), String> {
    with_frame_mut(id, |frame| {
        let incoming = values
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<HashSet<_>>();
        let removed = frame
            .state
            .names
            .iter()
            .filter_map(|(name, slot)| {
                (!incoming.contains(name.as_str())
                    && frame.global_aliases.binding_for_slot(*slot).is_none())
                .then_some(name.clone())
            })
            .collect::<Vec<_>>();
        for name in removed {
            remove_workspace_name(frame, &name);
        }
        for (name, value) in values {
            if frame
                .state
                .names
                .get(&name)
                .is_some_and(|slot| frame.global_aliases.binding_for_slot(*slot).is_some())
            {
                continue;
            }
            set_workspace_variable_in_frame(frame, &name, value)?;
        }
        Ok(())
    })
    .ok_or_else(|| "workspace frame is closed".to_string())?
}
