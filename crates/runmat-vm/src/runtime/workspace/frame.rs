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

pub(super) fn has_exact_pending_service_frame(owner: WorkspaceOwner) -> bool {
    let WorkspaceOwner::Runtime(_) = owner else {
        return false;
    };
    let Some(active) = active_frame(owner) else {
        return false;
    };
    let exact = PENDING_SERVICE_FRAMES.with(|frames| {
        frames
            .borrow()
            .get(&owner)
            .is_some_and(|pending| pending.iter().any(|frame| frame.id == active))
    });
    assert!(
        exact,
        "VM workspace entered without its exact reserved service frame"
    );
    true
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

pub(super) fn with_frame<R>(
    id: WorkspaceFrameId,
    f: impl FnOnce(&WorkspaceFrame) -> R,
) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let stacks = stacks.try_borrow().ok()?;
        stacks
            .values()
            .flat_map(|stack| stack.iter())
            .find(|frame| frame.id == id)
            .map(f)
    })
}

pub(super) fn with_frame_mut_unlocked<R>(
    id: WorkspaceFrameId,
    f: impl FnOnce(&mut WorkspaceFrame) -> R,
) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.try_borrow_mut().ok()?;
        stacks
            .values_mut()
            .flat_map(|stack| stack.iter_mut())
            .find(|frame| frame.id == id)
            .map(f)
    })
}

struct FrameDetachGuard(WorkspaceFrameId);

impl FrameDetachGuard {
    fn acquire(id: WorkspaceFrameId) -> Option<Self> {
        if with_frame(id, |frame| frame.state.mutation_phase.is_busy())? {
            return None;
        }
        DETACHING_WORKSPACE_FRAMES.with(|frames| frames.borrow_mut().insert(id).then_some(Self(id)))
    }
}

impl Drop for FrameDetachGuard {
    fn drop(&mut self) {
        DETACHING_WORKSPACE_FRAMES.with(|frames| {
            frames.borrow_mut().remove(&self.0);
        });
        let deferred = DEFERRED_WORKSPACE_FRAME_CLOSES.with(|pending| {
            let mut pending = pending.borrow_mut();
            let index = pending.iter().position(|(_, id)| *id == self.0)?;
            Some(pending.remove(index))
        });
        if let Some((owner, frame)) = deferred {
            drop(WorkspaceStateGuard { owner, frame });
        }
    }
}

pub(crate) fn workspace_frame_lookup(id: WorkspaceFrameId, name: &str) -> Option<Value> {
    with_frame(id, |frame| {
        let index = frame.state.names.get(name)?;
        frame
            .state
            .assigned
            .contains(name)
            .then(|| frame.state.values.get(*index).map(|value| value.clone()))
            .flatten()
    })
    .flatten()
}

pub(crate) fn workspace_frame_snapshot(id: WorkspaceFrameId) -> Option<Vec<(String, Value)>> {
    with_frame(id, snapshot_frame_entries)
}

fn snapshot_frame_entries(frame: &WorkspaceFrame) -> Vec<(String, Value)> {
    let mut entries = frame
        .state
        .names
        .iter()
        .filter_map(|(name, index)| {
            frame
                .state
                .assigned
                .contains(name)
                .then(|| frame.state.values.get(*index).map(|value| value.clone()))
                .flatten()
                .map(|value| (name.as_str().to_owned(), value))
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    entries
}

/// Hold only this frame's write authority across canonical publication. The
/// global stack borrow is released so a nested top-level VM delegate can
/// detach its different frame; any attempted reentry into this frame fails
/// closed rather than replacing a binding that this operation would erase.
pub(crate) fn workspace_frame_detach_selected(
    id: WorkspaceFrameId,
    selection: &runmat_runtime::workspace::WorkspaceDetachSelection,
    publish: impl FnOnce(
        &runmat_runtime::workspace::DetachedWorkspaceValues,
    ) -> Result<(), runmat_runtime::RuntimeError>,
) -> Result<runmat_runtime::workspace::DetachedWorkspaceValues, runmat_runtime::RuntimeError> {
    let unavailable = || {
        runmat_runtime::runtime_error::semantic_error(
            "RunMat:Workspace:FrameUnavailable",
            "workspace frame is closed or a frame transaction is already active",
        )
    };
    let _guard = FrameDetachGuard::acquire(id).ok_or_else(unavailable)?;
    let detached = with_frame(id, |frame| {
        runmat_runtime::workspace::DetachedWorkspaceValues::new(
            selection.select(snapshot_frame_entries(frame)),
        )
    })
    .ok_or_else(unavailable)?
    .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?;
    let context = runmat_runtime::context::legacy::active();
    let prepared = super::frame_lifecycle::prepare_workspace_frame_detach(
        id,
        context.as_ref(),
        selection,
        &detached,
    )?;
    prepared.checkpoint_before_publication()?;
    publish(&detached)?;
    prepared.commit();
    Ok(detached)
}

pub(crate) fn workspace_frame_publishes(id: WorkspaceFrameId) -> Option<bool> {
    with_frame(id, |frame| frame.policy.publishes_mutations())
}

/// Resolve the actual declared global authority for one workspace name.
/// The compiler/workspace spelling can differ from a named global alias;
/// callers must not manufacture the session target from `name`.
pub(crate) fn workspace_frame_global_binding(
    id: WorkspaceFrameId,
    name: &str,
) -> Option<(usize, runmat_runtime::workspace::GlobalBindingTarget)> {
    with_frame(id, |frame| {
        let slot = *frame.state.names.get(name)?;
        frame
            .global_aliases
            .binding_handle_for_slot(slot)
            .map(|binding| (slot, binding))
    })
    .flatten()
}

/// Resolve one compiler-owned lexical binding from the frame's canonical
/// name/slot tables. The returned environment handle and binding identity can
/// be retained across preparation without holding the workspace-stack borrow.
pub(crate) fn workspace_frame_lexical_binding(
    id: WorkspaceFrameId,
    name: &str,
) -> Option<(
    usize,
    runmat_value::LexicalEnvironment,
    crate::bytecode::program::LexicalBindingSlot,
)> {
    with_frame(id, |frame| {
        let slot = *frame.state.names.get(name)?;
        let lexical = frame.lexical.as_ref()?;
        lexical
            .slots
            .get(&slot)
            .copied()
            .map(|binding| (slot, lexical.environment.clone(), binding))
    })
    .flatten()
}

pub(crate) fn prepare_workspace_frame_declare_global(
    id: WorkspaceFrameId,
    context: &runmat_runtime::context::RuntimeContext,
    slot: usize,
    name: &str,
) -> Result<
    Option<runmat_runtime::workspace::bindings::PreparedAliasBinding>,
    runmat_runtime::RuntimeError,
> {
    let Some(aliases) = with_frame(id, |frame| frame.global_aliases.clone()) else {
        return Ok(None);
    };
    aliases
        .prepare_named(context, slot, name)
        .map(Some)
        .map_err(super::frame_allocation_error)
}

pub(crate) fn prepare_workspace_frame_declare_legacy_global(
    id: WorkspaceFrameId,
    context: &runmat_runtime::context::RuntimeContext,
    slot: usize,
) -> Result<
    Option<runmat_runtime::workspace::bindings::PreparedAliasBinding>,
    runmat_runtime::RuntimeError,
> {
    let Some(aliases) = with_frame(id, |frame| frame.global_aliases.clone()) else {
        return Ok(None);
    };
    aliases
        .prepare_legacy_slot(context, slot)
        .map(Some)
        .map_err(super::frame_allocation_error)
}

#[cfg(test)]
pub(crate) fn workspace_frame_assign(
    id: WorkspaceFrameId,
    name: &str,
    value: Value,
) -> Result<(), String> {
    let context = runmat_runtime::context::legacy::active();
    let prepared = prepare_workspace_frame_assignment(id, context.as_ref(), name, &value, true)
        .map_err(|error| error.to_string())?;
    prepared
        .checkpoint_before_publication()
        .map_err(|error| error.to_string())?;
    prepared.commit();
    Ok(())
}

#[cfg(test)]
pub(crate) fn workspace_frame_assign_from_executor(
    id: WorkspaceFrameId,
    name: &str,
    value: Value,
) -> Result<(), String> {
    let context = runmat_runtime::context::legacy::active();
    let prepared = prepare_workspace_frame_assignment(id, context.as_ref(), name, &value, false)
        .map_err(|error| error.to_string())?;
    prepared
        .checkpoint_before_publication()
        .map_err(|error| error.to_string())?;
    prepared.commit();
    Ok(())
}

#[cfg(test)]
pub(crate) fn workspace_frame_clear(id: WorkspaceFrameId) -> Result<(), String> {
    let context = runmat_runtime::context::legacy::active();
    let prepared =
        prepare_workspace_frame_clear(id, context.as_ref()).map_err(|error| error.to_string())?;
    prepared
        .checkpoint_before_publication()
        .map_err(|error| error.to_string())?;
    prepared.commit();
    Ok(())
}

#[cfg(test)]
pub(crate) fn reconcile_workspace_frame(
    id: WorkspaceFrameId,
    values: Vec<(String, Value)>,
) -> Result<(), String> {
    super::frame_reconcile::reconcile_workspace_frame(id, values)
}

pub(crate) fn reconcile_workspace_frame_with_context(
    id: WorkspaceFrameId,
    context: Option<&runmat_runtime::context::RuntimeContext>,
    values: Vec<(String, Value)>,
) -> Result<(), runmat_runtime::RuntimeError> {
    super::frame_reconcile::reconcile_workspace_frame_with_context(id, context, values)
}

#[cfg(test)]
mod reconciliation_tests {
    use super::*;

    #[test]
    fn canonical_refresh_preserves_unassigned_compiler_slot() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runtime_id = runtime.local_identity();
        let reserved = reserve_workspace_service_frame(runtime_id, None);
        let activation = frame_activation(runtime_id, reserved.id);
        futures::executor::block_on(activation.scope(runtime.scope(async {
            let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![
                Value::Num(41.0),
                Value::Num(9.0),
                Value::Num(0.0),
            ]);
            let _guard = set_workspace_state(
                HashMap::from([
                    ("warm".to_string(), 0),
                    ("removed".to_string(), 1),
                    ("later".to_string(), 2),
                ]),
                HashSet::from(["warm".to_string(), "removed".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let frame = with_stack(current_owner(), |stack| stack.last().unwrap().id);

            reconcile_workspace_frame(frame, vec![("warm".into(), Value::Num(41.0))]).unwrap();
            let (_, state) = capture_workspace_target(WorkspaceTarget::Current).unwrap();
            assert_eq!(state.names.get("later"), Some(&2));
            assert!(!state.assigned.contains("later"));
            assert!(!state.names.contains_key("removed"));

            workspace_frame_assign_from_executor(frame, "later", Value::Num(43.0)).unwrap();
            assert_eq!(
                workspace_frame_lookup(frame, "later"),
                Some(Value::Num(43.0))
            );
            assert_eq!(
                capture_workspace_target(WorkspaceTarget::Current)
                    .unwrap()
                    .1
                    .names
                    .get("later"),
                Some(&2)
            );
        })));
    }
}
