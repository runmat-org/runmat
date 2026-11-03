use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::workspace::bindings::{FrameGlobalBinding, FrameGlobalBindings};
use runmat_runtime::workspace::{GlobalBindingTarget, WorkspaceName};
use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

mod activation;
mod admission;
mod admitted_target;
mod completion;
mod declaration;
mod frame;
mod frame_assignment;
mod frame_lifecycle;
mod frame_metadata;
mod frame_reconcile;
mod lexical_error;
mod target_completion;
use activation::active_frame;
pub(crate) use activation::WorkspaceFrameActivation;
pub(crate) use activation::WorkspaceFrameFuture;
pub(crate) use admission::{
    detached_transactional_workspace_admission_with_global_bindings,
    detached_workspace_admission_with_global_bindings,
    transactional_workspace_admission_with_global_bindings,
};
pub use admission::{
    workspace_admission, workspace_admission_with_global_bindings, PendingWorkspaceAdmission,
    WorkspaceExecutionCompletion,
};
pub(crate) use admitted_target::AdmittedDynamicInput;
pub(crate) use declaration::prepare_workspace_persistent_slot;
use frame::{has_exact_pending_service_frame, take_service_frame, PendingServiceFrame};
pub(crate) use frame::{
    prepare_workspace_frame_declare_global, prepare_workspace_frame_declare_legacy_global,
};
pub(crate) use frame::{
    reconcile_workspace_frame_with_context, release_pending_workspace_service_frame,
    reserve_workspace_service_frame, workspace_frame_detach_selected, workspace_frame_for_service,
    workspace_frame_global_binding, workspace_frame_lexical_binding, workspace_frame_lookup,
    workspace_frame_publishes, workspace_frame_snapshot, WorkspaceFrameId, WorkspaceFrameLifecycle,
};
#[cfg(test)]
pub(crate) use frame::{
    workspace_frame_assign, workspace_frame_assign_from_executor, workspace_frame_clear,
};
#[cfg(test)]
pub(crate) use frame_assignment::prepare_workspace_frame_import;
pub(crate) use frame_assignment::{
    prepare_workspace_frame_assignment, prepare_workspace_frame_assignment_at_slot,
    prepare_workspace_frame_assignment_lease, prepare_workspace_frame_assignment_lease_at_slot,
    prepare_workspace_frame_import_lease,
};
pub(crate) use frame_lifecycle::{
    prepare_workspace_frame_clear, prepare_workspace_frame_remove, prepare_workspace_frame_unassign,
};
pub use frame_metadata::{FrameMap, FrameSet, FrameVec};
use lexical_error::{committed as committed_lexical_error, unavailable as workspace_unavailable};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WorkspaceOwner {
    Runtime(runmat_runtime::context::RuntimeContextLocalId),
    Ambient,
}

pub(crate) fn frame_activation(
    runtime: runmat_runtime::context::RuntimeContextLocalId,
    frame: WorkspaceFrameId,
) -> WorkspaceFrameActivation {
    WorkspaceFrameActivation::new(WorkspaceOwner::Runtime(runtime), frame)
}

fn current_owner() -> WorkspaceOwner {
    runmat_runtime::context::legacy::active()
        .map(|runtime| WorkspaceOwner::Runtime(runtime.local_identity()))
        .unwrap_or(WorkspaceOwner::Ambient)
}

#[derive(Debug, Clone)]
enum SlotLifecycle {
    Assigned(WorkspaceName),
    Unassigned(WorkspaceName),
}

impl SlotLifecycle {
    fn name(&self) -> &str {
        match self {
            SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name,
        }
    }

    fn is_assigned(&self) -> bool {
        matches!(self, SlotLifecycle::Assigned(_))
    }

    fn into_name(self) -> WorkspaceName {
        match self {
            Self::Assigned(name) | Self::Unassigned(name) => name,
        }
    }
}

struct WorkspaceState {
    names: FrameMap<WorkspaceName, usize>,
    assigned: FrameSet<WorkspaceName>,
    assigned_names_this_execution: FrameSet<WorkspaceName>,
    assigned_ids_this_execution: FrameSet<usize>,
    removed_slots_this_execution: FrameMap<usize, WorkspaceName>,
    removed_ids_this_execution: FrameSet<usize>,
    removed_names_this_execution: FrameSet<WorkspaceName>,
    slot_lifecycle: FrameMap<usize, SlotLifecycle>,
    values: InterpreterVars,
    pending_var_reconciliation: FrameSet<usize>,
    collection_gate: runmat_runtime::context::host_allocation::FrameCollectionGate,
    mutation_phase: frame_metadata::FrameMutationPhase,
}

struct WorkspaceFrame {
    id: WorkspaceFrameId,
    parent: Option<WorkspaceFrameId>,
    base: WorkspaceFrameId,
    lifecycle: Rc<Cell<WorkspaceFrameLifecycle>>,
    delegate: Rc<RefCell<Option<Rc<dyn runmat_runtime::context::RuntimeWorkspaceService>>>>,
    service: Option<Weak<dyn runmat_runtime::context::RuntimeWorkspaceService>>,
    session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
    lexical: Option<LexicalWorkspaceBindings>,
    state: WorkspaceState,
    global_aliases: FrameGlobalBindings,
    completion: Option<completion::CompletionHandle>,
    policy: WorkspaceFramePolicy,
    base_target: BaseTargetAuthority,
}

struct LexicalWorkspaceBindings {
    environment: runmat_value::LexicalEnvironment,
    slots: HashMap<usize, crate::bytecode::program::LexicalBindingSlot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceFramePolicy {
    /// A top-level execution publishes admitted mutations as they occur and
    /// retains its completed slot state for the host result adapter.
    TopLevel,
    /// Dynamic evaluation captures its completed slot state for an explicit
    /// commit by the caller, but never publishes mutations while evaluating.
    Transactional,
    /// Procedure-local frames neither publish nor become a host completion.
    Transient,
}

impl WorkspaceFramePolicy {
    fn publishes_mutations(self) -> bool {
        matches!(self, Self::TopLevel)
    }

    fn captures_completion(self) -> bool {
        matches!(self, Self::TopLevel | Self::Transactional)
    }
}

pub(crate) struct PendingWorkspace {
    pub(crate) input: PendingWorkspaceInput,
    pub(crate) policy: WorkspaceFramePolicy,
    pub(crate) base_target: BaseTargetAuthority,
}

pub(crate) enum PendingWorkspaceInput {
    Legacy {
        names: HashMap<String, usize>,
        assigned: HashSet<String>,
        global_aliases: FrameGlobalBindings,
    },
    Admitted(admitted_target::AdmittedWorkspaceMetadata),
}

#[derive(Debug)]
pub struct WorkspaceValueSnapshot {
    pub vars: InterpreterVars,
    pub names: FrameMap<WorkspaceName, usize>,
    pub assigned: FrameSet<WorkspaceName>,
    pub global_aliases: FrameGlobalBindings,
    /// The moved collections' issuer must survive the child frame and be
    /// adopted by the target before any subsequent growth of their backing.
    pub(crate) collection_gate: runmat_runtime::context::host_allocation::FrameCollectionGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceTarget {
    Current,
    Caller,
    Base,
}

/// Whether `Base` is an installed VM frame or an explicitly selected session
/// owner. This provenance is issued with the exact workspace admission, not
/// inferred from the presence of a runtime service or a builtin name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BaseTargetAuthority {
    Frame,
    CanonicalService,
}

pub(crate) fn base_target_authority() -> Option<BaseTargetAuthority> {
    let owner = current_owner();
    with_stack(owner, |stack| {
        let mut frame = active_frame(owner).or_else(|| (stack.len() == 1).then(|| stack[0].id))?;
        loop {
            let current = stack.iter().find(|candidate| candidate.id == frame)?;
            if current.base_target == BaseTargetAuthority::CanonicalService {
                return Some(BaseTargetAuthority::CanonicalService);
            }
            match current.parent {
                Some(parent) => frame = parent,
                None => return Some(BaseTargetAuthority::Frame),
            }
        }
    })
}

#[derive(Debug)]
pub struct WorkspaceTargetSnapshot {
    pub names: HashMap<String, usize>,
    pub assigned: HashSet<String>,
    pub vars: InterpreterVars,
    pub global_aliases: HashMap<usize, FrameGlobalBinding>,
}

/// The exact frame selected by CURRENT/CALLER/BASE before a VM request is
/// queued. The sibling uses the original current frame as its poll-scoped
/// activation, and the parent must still address this same target at commit.
#[derive(Clone, Copy)]
pub(crate) struct WorkspaceTargetIdentity {
    owner: WorkspaceOwner,
    current: WorkspaceFrameId,
    target: WorkspaceFrameId,
    selector: WorkspaceTarget,
    publication: Option<runmat_runtime::workspace::WorkspaceCompletionIdentity>,
}

impl WorkspaceTargetIdentity {
    pub(crate) fn ensure_current(
        self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        if WorkspaceOwner::Runtime(context.local_identity()) != self.owner
            || active_frame(self.owner) != Some(self.current)
            || target_frame_id(self.owner, self.selector) != Some(self.target)
        {
            return Err(workspace_target_error(
                "captured workspace target has changed",
            ));
        }
        Ok(())
    }

    pub(crate) fn scope<F: std::future::Future>(
        self,
        future: F,
    ) -> activation::WorkspaceFrameFuture<F> {
        WorkspaceFrameActivation::new(self.owner, self.current).scope(future)
    }

    pub(crate) fn commit(
        self,
        context: &runmat_runtime::context::RuntimeContext,
        snapshot: WorkspaceValueSnapshot,
        report: WorkspaceAssignedReport,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_current(context)?;
        replace_workspace_target_with_identity(
            Some(context),
            self.owner,
            self.selector,
            self.publication,
            snapshot,
            report,
        )
    }
}

#[derive(Debug)]
pub struct WorkspaceAssignedReport {
    pub ids: FrameSet<usize>,
    pub names: FrameSet<WorkspaceName>,
    pub removed_ids: FrameSet<usize>,
    pub removed_names: FrameSet<WorkspaceName>,
}

impl Default for WorkspaceAssignedReport {
    fn default() -> Self {
        let gate = runmat_runtime::context::host_allocation::FrameCollectionGate::unbounded();
        let allocator = gate.allocator();
        Self {
            ids: frame_metadata::empty_set(allocator.clone()),
            names: frame_metadata::empty_set(allocator.clone()),
            removed_ids: frame_metadata::empty_set(allocator.clone()),
            removed_names: frame_metadata::empty_set(allocator),
        }
    }
}

runmat_thread_local! {
    static WORKSPACE_STACKS: RefCell<HashMap<WorkspaceOwner, Vec<WorkspaceFrame>>> = RefCell::new(HashMap::new());
    static PENDING_SERVICE_FRAMES: RefCell<HashMap<WorkspaceOwner, Vec<PendingServiceFrame>>> = RefCell::new(HashMap::new());
    static NEXT_WORKSPACE_FRAME_ID: Cell<usize> = const { Cell::new(1) };
    static DETACHING_WORKSPACE_FRAMES: RefCell<HashSet<WorkspaceFrameId>> = RefCell::new(HashSet::new());
    static DEFERRED_WORKSPACE_FRAME_CLOSES: RefCell<Vec<(WorkspaceOwner, WorkspaceFrameId)>> = RefCell::new(Vec::new());
}

pub(crate) fn frame_is_detaching(id: WorkspaceFrameId) -> bool {
    DETACHING_WORKSPACE_FRAMES.with(|frames| frames.borrow().contains(&id))
}

fn with_stack<R>(owner: WorkspaceOwner, f: impl FnOnce(&[WorkspaceFrame]) -> R) -> R {
    WORKSPACE_STACKS.with(|stacks| {
        let stacks = stacks.borrow();
        let stack = stacks.get(&owner).map(Vec::as_slice).unwrap_or_default();
        let active = active_frame(owner)
            .and_then(|id| stack.iter().position(|frame| frame.id == id))
            .map(|index| &stack[..=index])
            .unwrap_or(stack);
        f(active)
    })
}

fn with_owner_stack_mut<R>(
    owner: WorkspaceOwner,
    f: impl FnOnce(&mut Vec<WorkspaceFrame>) -> R,
) -> R {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.borrow_mut();
        let result = f(stacks.entry(owner).or_default());
        if stacks.get(&owner).is_some_and(Vec::is_empty) {
            stacks.remove(&owner);
        }
        result
    })
}

fn with_active_frame_mut<R>(
    owner: WorkspaceOwner,
    operation: impl FnOnce(&mut WorkspaceFrame) -> R,
) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let mut stacks = stacks.borrow_mut();
        let stack = stacks.get_mut(&owner)?;
        let active = active_frame(owner).or_else(|| (stack.len() == 1).then(|| stack[0].id))?;
        let frame = stack.iter_mut().find(|frame| frame.id == active)?;
        if frame_is_detaching(frame.id) || frame.state.mutation_phase.is_busy() {
            return None;
        }
        Some(operation(frame))
    })
}

fn with_frame_by_id<R>(
    owner: WorkspaceOwner,
    id: WorkspaceFrameId,
    operation: impl FnOnce(&WorkspaceFrame) -> R,
) -> Option<R> {
    WORKSPACE_STACKS.with(|stacks| {
        let stacks = stacks.borrow();
        stacks
            .get(&owner)?
            .iter()
            .find(|frame| frame.id == id)
            .map(operation)
    })
}

/// Update removal metadata without destroying a displaced shared name while
/// the workspace stack is borrowed. The caller owns retirement ordering.
fn mark_slot_unassigned_retaining(
    ws: &mut WorkspaceState,
    index: usize,
    name: WorkspaceName,
) -> [Option<WorkspaceName>; 2] {
    let mut retired = [None, None];
    if let Some(entry) = ws.slot_lifecycle.get_mut(&index) {
        retired[0] =
            Some(std::mem::replace(entry, SlotLifecycle::Unassigned(name.clone())).into_name());
    } else {
        ws.slot_lifecycle
            .insert(index, SlotLifecycle::Unassigned(name.clone()));
    }
    if let Some(entry) = ws.removed_slots_this_execution.get_mut(&index) {
        retired[1] = Some(std::mem::replace(entry, name));
    } else {
        ws.removed_slots_this_execution.insert(index, name);
    }
    if !ws.removed_ids_this_execution.contains(&index) {
        ws.removed_ids_this_execution.insert(index);
    }
    if let Some(name) = ws.removed_slots_this_execution.get(&index) {
        if !ws.removed_names_this_execution.contains(name) {
            ws.removed_names_this_execution.insert(name.clone());
        }
    }
    retired
}

/// Update assignment metadata and return every displaced shared name. At
/// most one lifecycle name, one removed-slot name, and one removed-name key
/// can be displaced by a single slot assignment.
fn mark_slot_assigned_retaining(
    ws: &mut WorkspaceState,
    index: usize,
    name: &WorkspaceName,
) -> [Option<WorkspaceName>; 3] {
    let mut retired = [None, None, None];
    if let Some(slot) = ws.slot_lifecycle.get_mut(&index) {
        if slot.name() != name.as_str() || matches!(slot, SlotLifecycle::Unassigned(_)) {
            retired[0] =
                Some(std::mem::replace(slot, SlotLifecycle::Assigned(name.clone())).into_name());
        }
    } else {
        ws.slot_lifecycle
            .insert(index, SlotLifecycle::Assigned(name.clone()));
    }
    retired[1] = ws.removed_slots_this_execution.remove(&index);
    ws.removed_ids_this_execution.remove(&index);
    retired[2] = ws.removed_names_this_execution.take(name.as_str());
    retired
}

fn find_unassigned_slot_for_name(ws: &WorkspaceState, name: &str) -> Option<usize> {
    ws.slot_lifecycle.iter().find_map(|(idx, state)| {
        matches!(state, SlotLifecycle::Unassigned(slot_name) if slot_name.as_str() == name)
            .then_some(*idx)
    })
}

fn target_frame_id(owner: WorkspaceOwner, target: WorkspaceTarget) -> Option<WorkspaceFrameId> {
    with_stack(owner, |stack| {
        let current = active_frame(owner).or_else(|| (stack.len() == 1).then(|| stack[0].id))?;
        let frame = stack.iter().find(|frame| frame.id == current)?;
        Some(match target {
            WorkspaceTarget::Current => current,
            WorkspaceTarget::Caller => frame.parent.unwrap_or(current),
            WorkspaceTarget::Base => frame.base,
        })
    })
}

pub struct WorkspaceStateGuard {
    owner: WorkspaceOwner,
    frame: WorkspaceFrameId,
}

impl Drop for WorkspaceStateGuard {
    fn drop(&mut self) {
        if let Some(phase) = with_frame_by_id(self.owner, self.frame, |frame| {
            frame.state.mutation_phase.clone()
        }) {
            if phase.is_busy() {
                phase.defer_close();
                return;
            }
        }
        if frame_is_detaching(self.frame) {
            DEFERRED_WORKSPACE_FRAME_CLOSES.with(|pending| {
                pending.borrow_mut().push((self.owner, self.frame));
            });
            return;
        }
        let Some(mut frame) = with_owner_stack_mut(self.owner, |stack| {
            stack
                .iter()
                .position(|frame| frame.id == self.frame)
                .map(|index| stack.remove(index))
        }) else {
            return;
        };
        frame.lifecycle.set(WorkspaceFrameLifecycle::Closed);
        let retired_delegate = frame.delegate.borrow_mut().take();
        if frame.policy.captures_completion() {
            let ws = frame.state;
            let assigned = WorkspaceAssignedReport {
                ids: ws.assigned_ids_this_execution,
                names: ws.assigned_names_this_execution,
                removed_ids: ws.removed_ids_this_execution,
                removed_names: ws.removed_names_this_execution,
            };
            let state = WorkspaceValueSnapshot {
                vars: ws.values,
                names: ws.names,
                assigned: ws.assigned,
                global_aliases: frame.global_aliases,
                collection_gate: ws.collection_gate,
            };
            frame
                .completion
                .take()
                .expect("capturing frame has a prepared completion node")
                .publish(state, assigned);
        }
        drop(retired_delegate);
    }
}

pub(crate) fn take_pending_workspace_state() -> Option<PendingWorkspace> {
    frame::take_service_frame_admission(current_owner())
}

pub(crate) fn has_exact_pending_workspace_service_frame() -> bool {
    has_exact_pending_service_frame(current_owner())
}

pub fn take_updated_workspace_state() -> Option<WorkspaceValueSnapshot> {
    completion::take_state_for_owner(current_owner())
}

pub fn take_updated_workspace_assigned_report() -> Option<WorkspaceAssignedReport> {
    completion::take_assigned_for_owner(current_owner())
}

pub(crate) fn take_workspace_completion(
    frame: WorkspaceFrameId,
) -> Option<WorkspaceExecutionCompletion> {
    let (state, assigned) = completion::take_for_frame(frame)?;
    match (state, assigned) {
        (Some(state), Some(assigned)) => Some(WorkspaceExecutionCompletion { state, assigned }),
        (None, None) => None,
        (state, assigned) => {
            debug_assert!(false, "workspace completion was only partially recorded");
            state.map(|state| WorkspaceExecutionCompletion {
                state,
                assigned: assigned.unwrap_or_default(),
            })
        }
    }
}

#[cfg(test)]
pub(crate) fn set_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut InterpreterVars,
    global_aliases: FrameGlobalBindings,
) -> WorkspaceStateGuard {
    set_workspace_state_with_policy(
        names,
        assigned,
        vars,
        WorkspaceFramePolicy::TopLevel,
        BaseTargetAuthority::Frame,
        global_aliases,
    )
}

pub fn try_set_transient_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut InterpreterVars,
    global_aliases: FrameGlobalBindings,
) -> Result<WorkspaceStateGuard, runmat_runtime::RuntimeError> {
    try_set_workspace_state_with_policy(
        names,
        assigned,
        vars,
        WorkspaceFramePolicy::Transient,
        BaseTargetAuthority::Frame,
        global_aliases,
    )
}

#[cfg(test)]
pub(crate) fn set_transient_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut InterpreterVars,
    global_aliases: FrameGlobalBindings,
) -> WorkspaceStateGuard {
    install_workspace_state(
        names,
        assigned,
        vars,
        WorkspaceFramePolicy::Transient,
        BaseTargetAuthority::Frame,
        global_aliases,
    )
    .expect("test transient frame admission")
}

fn install_workspace_state(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut InterpreterVars,
    policy: WorkspaceFramePolicy,
    base_target: BaseTargetAuthority,
    global_aliases: FrameGlobalBindings,
) -> Result<WorkspaceStateGuard, runmat_runtime::RuntimeError> {
    use runmat_runtime::context::host_allocation::HostAllocationReceipt;
    let context = runmat_runtime::context::legacy::active();
    let owner = current_owner();
    let plan = frame_metadata::initial_plan(
        &names,
        &assigned,
        &global_aliases,
        vars.len(),
        policy.captures_completion(),
    )
    .map_err(frame_allocation_error)?;
    let receipt =
        HostAllocationReceipt::admit(context.as_ref(), plan).map_err(frame_allocation_error)?;
    if let (Some(context), Some(receipt)) = (context.as_ref(), receipt.as_ref()) {
        global_aliases
            .admit_ingress(context, receipt)
            .map_err(frame_allocation_error)?;
    }
    let state =
        frame_metadata::build_initial(names, assigned, vars, context.as_ref(), receipt.as_ref())
            .map_err(frame_allocation_error)?;
    install_prepared_workspace_state(
        owner,
        context.as_ref(),
        receipt.as_ref(),
        state,
        global_aliases,
        policy,
        base_target,
    )
}

fn install_prepared_workspace_state(
    owner: WorkspaceOwner,
    context: Option<&runmat_runtime::context::RuntimeContext>,
    receipt: Option<&runmat_runtime::context::host_allocation::HostAllocationReceipt>,
    state: WorkspaceState,
    global_aliases: FrameGlobalBindings,
    policy: WorkspaceFramePolicy,
    base_target: BaseTargetAuthority,
) -> Result<WorkspaceStateGuard, runmat_runtime::RuntimeError> {
    let completion = policy
        .captures_completion()
        .then(|| completion::CompletionHandle::prepare(owner, context, receipt))
        .transpose()
        .map_err(frame_allocation_error)?;
    let service_frame = take_service_frame(owner);
    service_frame.lifecycle.set(WorkspaceFrameLifecycle::Active);
    let frame_id = service_frame.id;
    if let Some(completion) = &completion {
        completion.bind_frame(frame_id);
    }
    let parent = service_frame.parent.get();
    let base = service_frame.base.get().unwrap_or(frame_id);
    let service = service_frame.service.borrow().clone();
    let session_variables = service_frame.session_variables.borrow().clone();
    with_owner_stack_mut(owner, |stack| {
        stack.push(WorkspaceFrame {
            id: service_frame.id,
            parent,
            base,
            lifecycle: service_frame.lifecycle,
            delegate: service_frame.delegate,
            service,
            session_variables,
            lexical: None,
            global_aliases,
            completion,
            state,
            policy,
            base_target,
        });
    });
    Ok(WorkspaceStateGuard {
        owner,
        frame: frame_id,
    })
}

pub(crate) fn install_admitted_workspace_state(
    metadata: admitted_target::AdmittedWorkspaceMetadata,
    vars: &mut InterpreterVars,
    policy: WorkspaceFramePolicy,
    base_target: BaseTargetAuthority,
) -> Result<WorkspaceStateGuard, runmat_runtime::RuntimeError> {
    let context = runmat_runtime::context::legacy::active()
        .ok_or_else(|| workspace_target_error("admitted workspace has no active runtime"))?;
    let owner = WorkspaceOwner::Runtime(context.local_identity());
    let aliases = metadata.aliases.clone();
    let (state, receipt) = admitted_target::build_frame_state(metadata, vars, &context, policy)?;
    install_prepared_workspace_state(
        owner,
        Some(&context),
        receipt.as_ref(),
        state,
        aliases,
        policy,
        base_target,
    )
}

pub(crate) fn frame_allocation_error(
    error: runmat_runtime::context::host_allocation::HostAllocationError,
) -> runmat_runtime::RuntimeError {
    let runtime = runmat_runtime::context::legacy::active();
    frame_allocation_error_in_context(runtime.as_ref(), error)
}

pub(crate) fn frame_allocation_error_in_context(
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
    error: runmat_runtime::context::host_allocation::HostAllocationError,
) -> runmat_runtime::RuntimeError {
    let identifier = format!("RunMat:Workspace:{}", error.runtime_identifier_suffix());
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        runtime,
        &identifier,
        "workspace frame metadata admission failed",
        error,
    )
}

/// Install the selected function's typed lexical slots on its existing
/// workspace frame. Clear/remove then retire the same binding cells after the
/// workspace effect has committed, without a second name resolver.
pub(crate) fn attach_lexical_workspace_bindings(
    environment: Option<&runmat_value::LexicalEnvironment>,
    bindings: &[crate::bytecode::program::LexicalBindingSlot],
) -> Result<(), String> {
    if bindings.is_empty() {
        return Ok(());
    }
    let environment = environment
        .ok_or_else(|| "lexical workspace bindings have no activation".to_string())?
        .clone();
    let mut slots = HashMap::new();
    for binding in bindings {
        if slots.insert(binding.slot, *binding).is_some() {
            return Err("lexical workspace has a duplicate binding slot".into());
        }
    }
    with_active_frame_mut(current_owner(), |frame| {
        if frame.lexical.is_some() {
            return Err("lexical workspace bindings were installed twice".into());
        }
        frame.lexical = Some(LexicalWorkspaceBindings { environment, slots });
        Ok(())
    })
    .unwrap_or_else(|| Err("lexical workspace frame is unavailable".into()))
}

pub(crate) fn try_set_workspace_state_with_policy(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut InterpreterVars,
    policy: WorkspaceFramePolicy,
    base_target: BaseTargetAuthority,
    global_aliases: FrameGlobalBindings,
) -> Result<WorkspaceStateGuard, runmat_runtime::RuntimeError> {
    install_workspace_state(names, assigned, vars, policy, base_target, global_aliases)
}

#[cfg(test)]
pub(crate) fn set_workspace_state_with_policy(
    names: HashMap<String, usize>,
    assigned: HashSet<String>,
    vars: &mut InterpreterVars,
    policy: WorkspaceFramePolicy,
    base_target: BaseTargetAuthority,
    global_aliases: FrameGlobalBindings,
) -> WorkspaceStateGuard {
    install_workspace_state(names, assigned, vars, policy, base_target, global_aliases)
        .expect("test workspace frame admission")
}

fn refresh_workspace_slot_owned(index: usize, value: Value) {
    let retired = with_active_frame_mut(current_owner(), |frame| {
        if frame.state.values.len() <= index {
            frame.state.values.resize_owned(index + 1, Value::Num(0.0));
        }
        frame.state.values.replace_owned(index, value)
    });
    drop(retired);
}

/// Refresh a compatibility frame from an interpreter slot without holding a
/// shared-control read guard while the frame is mutated. Canonical frames
/// already share this exact control and need no mirror write or Value clone.
pub fn refresh_workspace_slot_from_vars(index: usize, vars: &InterpreterVars) {
    let already_shared = with_active_frame_mut(current_owner(), |frame| {
        frame.state.values.shares_storage_with(vars)
    })
    .unwrap_or(false);
    if already_shared {
        return;
    }
    let Some(value) = vars.get(index).map(|value| value.clone()) else {
        return;
    };
    refresh_workspace_slot_owned(index, value);
}

pub fn reconcile_workspace_vars(vars: &mut InterpreterVars) {
    let authoritative = with_active_frame_mut(current_owner(), |frame| {
        // This runs before every instruction. Avoid touching the shared
        // control's refcount unless a completed child actually replaced it.
        let authoritative = frame.state.values.shared_clone_if_distinct(vars);
        frame.state.pending_var_reconciliation.clear();
        authoritative
    });
    let Some(Some(authoritative)) = authoritative else {
        return;
    };
    let retired = vars.adopt_shared(authoritative);
    // A completed child can replace the whole slot generation. Retire the
    // parent's former control only after the frame-stack borrow is released.
    drop(retired);
}

pub fn workspace_lookup(name: &str) -> Option<Value> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let ws = &frame.state;
        let idx = ws.names.get(name)?;
        if !ws.assigned.contains(name) {
            return None;
        }
        ws.values.get(*idx).map(|value| value.clone())
    })
}

pub fn workspace_slot_assigned(index: usize) -> Option<bool> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let ws = &frame.state;
        ws.slot_lifecycle
            .get(&index)
            .map(SlotLifecycle::is_assigned)
    })
}

pub(crate) fn current_workspace_assigned_slot(name: &str) -> Option<usize> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let slot = *frame.state.names.get(name)?;
        frame
            .state
            .slot_lifecycle
            .get(&slot)?
            .is_assigned()
            .then_some(slot)
    })
}

pub fn workspace_slot_name(index: usize) -> Option<String> {
    workspace_slot_name_handle(index).map(|name| name.as_str().to_owned())
}

/// Retain the existing admitted name across a service call without copying
/// its text or holding a borrow of the workspace stack.
pub(crate) fn workspace_slot_name_handle(index: usize) -> Option<WorkspaceName> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let ws = &frame.state;
        ws.slot_lifecycle.get(&index).map(|state| match state {
            SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name.clone(),
        })
    })
}

/// Binding metadata for deferred closure issuance. The value extent includes
/// unnamed frame slots; no Values or second workspace snapshot are copied.
pub(crate) struct WorkspaceBindingState {
    pub(crate) frame: WorkspaceFrameId,
    pub(crate) names: HashMap<String, usize>,
    pub(crate) assigned: HashSet<String>,
    pub(crate) value_count: usize,
}

pub(crate) fn workspace_binding_state() -> Option<WorkspaceBindingState> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        Some(WorkspaceBindingState {
            frame: frame.id,
            names: frame
                .state
                .names
                .iter()
                .map(|(name, slot)| (name.as_str().to_owned(), *slot))
                .collect(),
            assigned: frame
                .state
                .assigned
                .iter()
                .map(|name| name.as_str().to_owned())
                .collect(),
            value_count: frame.state.values.len(),
        })
    })
}

/// The active frame identity without copying workspace names or Values.
pub(crate) fn workspace_frame_identity() -> Option<WorkspaceFrameId> {
    let owner = current_owner();
    with_stack(owner, |stack| {
        active_frame(owner).or_else(|| (stack.len() == 1).then(|| stack[0].id))
    })
}

pub fn workspace_assign(name: &str, value: Value) -> Result<(), runmat_runtime::RuntimeError> {
    workspace_assign_target(WorkspaceTarget::Current, name, value)
}

pub fn workspace_assign_target(
    target: WorkspaceTarget,
    name: &str,
    value: Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    let owner = current_owner();
    let frame = target_frame_id(owner, target)
        .ok_or_else(|| workspace_unavailable("load: workspace state unavailable"))?;
    let context = runmat_runtime::context::legacy::active();
    let mut owned = Some(value);
    let retained = context
        .as_ref()
        .map(|context| {
            runmat_runtime::workspace::WorkspaceSlotReadLease::prepare_owned(
                owned
                    .take()
                    .expect("workspace value is moved into one retained generation"),
                context,
            )
        })
        .transpose()?;
    // Explicitly unbounded compatibility calls retain the previous rooted
    // owner. Operation-bearing writes move the exact admitted generation into
    // the frame token and never copy the Value payload.
    let admitted = owned
        .take()
        .map(runmat_runtime::execution::RootedValue::new)
        .transpose()
        .map_err(|error| error.into_runtime_error())?;
    let value = retained
        .as_ref()
        .map(runmat_runtime::workspace::WorkspaceSlotReadLease::value)
        .or_else(|| {
            admitted
                .as_ref()
                .map(runmat_runtime::execution::RootedValue::value)
        })
        .expect("workspace assignment retains one source owner");
    let prepared = match (context.as_ref(), retained.clone()) {
        (Some(context), Some(retained)) => {
            prepare_workspace_frame_assignment_lease(frame, context, name, retained, true)?
        }
        _ => prepare_workspace_frame_assignment(frame, context.as_ref(), name, value, true)?,
    };
    let slot = prepared.slot();
    let shared_value = value.clone();
    let (global, lexical) = with_frame_by_id(owner, frame, |frame| {
        let global = frame.global_aliases.binding_handle_for_slot(slot);
        let lexical = frame.lexical.as_ref().and_then(|owner| {
            owner
                .slots
                .get(&slot)
                .copied()
                .map(|binding| (owner.environment.clone(), binding))
        });
        (global, lexical)
    })
    .ok_or_else(|| workspace_unavailable("load: workspace state unavailable"))?;
    let session = with_frame_by_id(owner, frame, |frame| frame.session_variables.clone()).flatten();
    let stores_global = global.is_some();
    if stores_global && context.is_some() && session.is_none() {
        return Err(workspace_unavailable(
            "workspace frame has no session-variable authority",
        ));
    }
    #[cfg(not(test))]
    if stores_global && context.is_none() {
        return Err(workspace_unavailable(
            "workspace frame has no session-variable authority",
        ));
    }
    let global_write = match (context.as_ref(), session.as_ref(), global.as_ref()) {
        (Some(context), Some(session), Some(GlobalBindingTarget::Named(bound))) => Some(
            session.prepare_store_global_with_context(context, bound.as_str(), value.clone())?,
        ),
        (Some(context), Some(session), Some(GlobalBindingTarget::LegacySlot(bound))) => {
            Some(session.prepare_store_global_slot_with_context(context, *bound, value.clone())?)
        }
        _ => None,
    };
    prepared.checkpoint_before_publication()?;
    if let (Some(context), Some(write)) = (context.as_ref(), global_write.as_ref()) {
        write.checkpoint(context)?;
    }
    prepared.commit();
    if let Some(write) = global_write {
        write.commit();
    }
    #[cfg(test)]
    if context.is_none() {
        match global.as_ref() {
            Some(GlobalBindingTarget::Named(bound)) => {
                runmat_runtime::workspace::session::store_global_named(
                    bound.as_str(),
                    value.clone(),
                )?;
            }
            Some(GlobalBindingTarget::LegacySlot(bound)) => {
                runmat_runtime::workspace::session::store_global_slot(*bound, value.clone())?;
            }
            _ => {}
        }
    }
    if let Some((environment, binding)) = lexical {
        publish_lexical_binding(&environment, binding, shared_value)?;
    }
    Ok(())
}

/// The workspace assignment has committed before this shared-cell publication.
/// Do not hold the workspace stack borrow while mutating the GC-owned cell.
fn publish_lexical_binding(
    environment: &runmat_value::LexicalEnvironment,
    binding: crate::bytecode::program::LexicalBindingSlot,
    value: Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    let result = match binding.kind {
        crate::bytecode::program::LexicalBindingKind::Owned => {
            runmat_runtime::call::lexical::write_owned_binding(environment, binding.binding, value)
        }
        crate::bytecode::program::LexicalBindingKind::Inherited => {
            runmat_runtime::call::lexical::write_binding(environment, binding.binding, value)
        }
        crate::bytecode::program::LexicalBindingKind::Transit => return Ok(()),
    };
    result.map_err(|error| {
        committed_lexical_error(
            "RunMat:Workspace:LexicalPublication",
            format!("workspace assignment committed before lexical publication: {error}"),
            error,
        )
    })
}

pub fn workspace_clear() -> Result<(), runmat_runtime::RuntimeError> {
    let (frame, retired) = with_active_frame_mut(current_owner(), |frame| {
        let lexical = frame.lexical.as_ref().map(|owner| {
            (
                owner.environment.clone(),
                owner.slots.values().copied().collect::<Vec<_>>(),
            )
        });
        (frame.id, lexical)
    })
    .ok_or_else(|| workspace_unavailable("clear: workspace state unavailable"))?;
    let context = runmat_runtime::context::legacy::active();
    let prepared = frame_lifecycle::prepare_workspace_frame_clear(frame, context.as_ref())?;
    prepared.checkpoint_before_publication()?;
    prepared.commit();
    retire_lexical_bindings(retired)
}

pub fn workspace_remove(name: &str) -> Result<(), runmat_runtime::RuntimeError> {
    let (frame, retired) = with_active_frame_mut(current_owner(), |frame| {
        let lexical = frame.state.names.get(name).and_then(|slot| {
            frame.lexical.as_ref().and_then(|owner| {
                owner
                    .slots
                    .get(slot)
                    .copied()
                    .map(|binding| (owner.environment.clone(), vec![binding]))
            })
        });
        (frame.id, lexical)
    })
    .ok_or_else(|| workspace_unavailable("clear: workspace state unavailable"))?;
    let context = runmat_runtime::context::legacy::active();
    let prepared = frame_lifecycle::prepare_workspace_frame_remove(frame, context.as_ref(), name)?;
    prepared.checkpoint_before_publication()?;
    prepared.commit();
    retire_lexical_bindings(retired)
}

/// The workspace mutation has already committed when this runs. A failure is
/// reported as a post-commit retirement error, never as a fictitious rollback.
fn retire_lexical_bindings(
    retired: Option<(
        runmat_value::LexicalEnvironment,
        Vec<crate::bytecode::program::LexicalBindingSlot>,
    )>,
) -> Result<(), runmat_runtime::RuntimeError> {
    let Some((environment, bindings)) = retired else {
        return Ok(());
    };
    for binding in bindings {
        retire_lexical_binding(&environment, binding)?;
    }
    Ok(())
}

fn retire_lexical_binding(
    environment: &runmat_value::LexicalEnvironment,
    binding: crate::bytecode::program::LexicalBindingSlot,
) -> Result<(), runmat_runtime::RuntimeError> {
    let result = match binding.kind {
        crate::bytecode::program::LexicalBindingKind::Owned => {
            runmat_runtime::call::lexical::remove_owned_binding(environment, binding.binding)
        }
        crate::bytecode::program::LexicalBindingKind::Inherited => {
            runmat_runtime::call::lexical::remove_binding(environment, binding.binding)
        }
        crate::bytecode::program::LexicalBindingKind::Transit => return Ok(()),
    };
    result.map_err(|error| {
        committed_lexical_error(
            "RunMat:Workspace:LexicalRetirement",
            format!("workspace mutation committed before lexical retirement: {error}"),
            error,
        )
    })
}

pub fn workspace_snapshot() -> Vec<(String, Value)> {
    with_stack(current_owner(), |stack| {
        if let Some(frame) = stack.last() {
            let ws = &frame.state;
            let mut entries: Vec<(String, Value)> = ws
                .names
                .iter()
                .filter_map(|(name, idx)| {
                    if !ws.assigned.contains(name) {
                        return None;
                    }
                    ws.values
                        .get(*idx)
                        .map(|value| (name.as_str().to_owned(), value.clone()))
                })
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        } else {
            Vec::new()
        }
    })
}

#[cfg(test)]
pub(crate) fn capture_workspace_target(
    target: WorkspaceTarget,
) -> Result<(WorkspaceTargetIdentity, WorkspaceTargetSnapshot), String> {
    capture_workspace_target_for_owner(current_owner(), target, None)
}

pub(crate) fn capture_workspace_target_in_context(
    context: &runmat_runtime::context::RuntimeContext,
    target: WorkspaceTarget,
) -> Result<(WorkspaceTargetIdentity, WorkspaceTargetSnapshot), String> {
    capture_workspace_target_for_owner(
        WorkspaceOwner::Runtime(context.local_identity()),
        target,
        Some(context),
    )
}

fn capture_workspace_target_for_owner(
    owner: WorkspaceOwner,
    target: WorkspaceTarget,
    context: Option<&runmat_runtime::context::RuntimeContext>,
) -> Result<(WorkspaceTargetIdentity, WorkspaceTargetSnapshot), String> {
    let (current, frame) = with_stack(owner, |stack| {
        let current = active_frame(owner).or_else(|| (stack.len() == 1).then(|| stack[0].id))?;
        let current_frame = stack.iter().find(|frame| frame.id == current)?;
        Some((
            current,
            match target {
                WorkspaceTarget::Current => current,
                WorkspaceTarget::Caller => current_frame.parent.unwrap_or(current),
                WorkspaceTarget::Base => current_frame.base,
            },
        ))
    })
    .ok_or_else(|| "workspace state unavailable".to_string())?;
    if let Some(context) = context {
        // A canonical base mutation may have advanced the publishing frame's
        // delegate since its last VM operation. Synchronize through that exact
        // frame's existing service before capturing names, values and revision.
        // Never call the service while holding the workspace stack borrow.
        let service = with_frame_by_id(owner, frame, |frame| {
            frame.service.as_ref().and_then(std::rc::Weak::upgrade)
        })
        .flatten()
        .ok_or_else(|| "workspace target service unavailable".to_string())?;
        service
            .ensure_effect_authority_with_context(context)
            .map_err(|error| error.to_string())?;
    }
    let publication =
        capture_target_publication(owner, frame).map_err(|error| error.to_string())?;
    let snapshot = with_frame_by_id(owner, frame, |frame| WorkspaceTargetSnapshot {
        names: frame
            .state
            .names
            .iter()
            .map(|(name, slot)| (name.as_str().to_owned(), *slot))
            .collect(),
        assigned: frame
            .state
            .assigned
            .iter()
            .map(|name| name.as_str().to_owned())
            .collect(),
        vars: InterpreterVars::from_owned(frame.state.values.clone_values()),
        global_aliases: frame.global_aliases.snapshot(),
    })
    .ok_or_else(|| "workspace state unavailable".to_string())?;
    Ok((
        WorkspaceTargetIdentity {
            owner,
            current,
            target: frame,
            selector: target,
            publication,
        },
        snapshot,
    ))
}

/// A context-bound lookup must not treat an unrelated ambient VM frame as the
/// selected workspace target. Native execution normally has no VM frame.
pub(crate) fn has_workspace_target_in_context(
    context: &runmat_runtime::context::RuntimeContext,
    target: WorkspaceTarget,
) -> bool {
    let owner = WorkspaceOwner::Runtime(context.local_identity());
    target_frame_id(owner, target).is_some()
}

fn capture_target_publication(
    owner: WorkspaceOwner,
    frame: WorkspaceFrameId,
) -> Result<
    Option<runmat_runtime::workspace::WorkspaceCompletionIdentity>,
    runmat_runtime::RuntimeError,
> {
    let delegate = with_frame_by_id(owner, frame, |frame| {
        frame
            .policy
            .publishes_mutations()
            .then(|| frame.delegate.borrow().clone())
            .flatten()
    })
    .ok_or_else(|| workspace_target_error("workspace state unavailable"))?;
    match (delegate, runmat_runtime::context::legacy::active()) {
        (Some(delegate), Some(context)) => {
            delegate.capture_workspace_completion_identity_with_context(&context)
        }
        _ => Ok(None),
    }
}

fn replace_workspace_target_with_identity(
    context: Option<&runmat_runtime::context::RuntimeContext>,
    owner: WorkspaceOwner,
    target: WorkspaceTarget,
    publication: Option<runmat_runtime::workspace::WorkspaceCompletionIdentity>,
    snapshot: WorkspaceValueSnapshot,
    report: WorkspaceAssignedReport,
) -> Result<(), runmat_runtime::RuntimeError> {
    let frame = target_frame_id(owner, target)
        .ok_or_else(|| workspace_target_error("workspace state unavailable"))?;
    let delegate = with_frame_by_id(owner, frame, |frame| {
        frame
            .policy
            .publishes_mutations()
            .then(|| frame.delegate.borrow().clone())
            .flatten()
    })
    .ok_or_else(|| workspace_target_error("workspace state unavailable"))?;
    let prepared =
        target_completion::PreparedTargetCompletion::prepare(frame, context, snapshot, report)?;
    if let Some(delegate) = delegate {
        let context = context.ok_or_else(|| {
            workspace_target_error("workspace completion requires its runtime context")
        })?;
        let publication = publication.ok_or_else(|| {
            workspace_target_error("workspace delegate does not support atomic completion")
        })?;
        let base =
            delegate.prepare_workspace_completion_with_context(context, publication, &prepared)?;
        return prepared.commit_with_base(context, base);
    }
    if let Some(context) = context {
        return prepared.commit_without_base(context);
    }
    prepared.checkpoint_before_publication()?;
    prepared.commit()
}

fn workspace_target_error(message: &'static str) -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error("RunMat:Workspace:TargetState", message)
}

#[cfg(test)]
pub fn set_workspace_variable(
    name: &str,
    value: Value,
    vars: &mut InterpreterVars,
) -> Result<(), String> {
    workspace_assign(name, value).map_err(|error| error.to_string())?;
    if let Some(value) = workspace_lookup(name) {
        let index = with_stack(current_owner(), |stack| {
            stack
                .last()
                .and_then(|frame| frame.state.names.get(name).copied())
        })
        .ok_or_else(|| "load: workspace slot unavailable".to_string())?;
        if vars.len() <= index {
            vars.resize_owned(index + 1, Value::Num(0.0));
        }
        let retired = vars.replace_owned(index, value);
        drop(retired);
    }
    Ok(())
}

/// Synchronize an observed ancestor-cell value without treating the read as
/// another executor assignment or workspace publication.
#[cfg(test)]
pub(crate) fn refresh_workspace_shared_slot(
    index: usize,
    value: &Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    let Some(frame) = workspace_frame_identity() else {
        return Ok(());
    };
    let context = runmat_runtime::context::legacy::active();
    let prepared = prepare_workspace_frame_import(frame, context.as_ref(), index, value)?;
    prepared.checkpoint_before_publication()?;
    prepared.commit();
    Ok(())
}

/// Admit a language-visible named-slot write before changing the VM register
/// file. Installed VM workspace services update both the exact frame and, for
/// a top-level invocation, its Runtime-owned canonical projection.
pub fn admit_workspace_slot_write(
    index: usize,
    value: &Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    let name = workspace_slot_name_handle(index);
    let Some(name) = name else {
        return Ok(());
    };
    admit_named_workspace_write(index, name.as_str(), value)
}

fn compiler_workspace_slot_matches(index: usize, name: &str) -> bool {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let ws = &frame.state;
        Some(
            ws.names.get(name).is_none_or(|slot| *slot == index)
                && ws
                    .slot_lifecycle
                    .get(&index)
                    .is_none_or(|slot| slot.name() == name),
        )
    })
    .unwrap_or(false)
}

pub(crate) fn admit_workspace_slot_lease(
    index: usize,
    name: &str,
    value: runmat_runtime::workspace::WorkspaceSlotReadLease,
) -> Result<(), runmat_runtime::RuntimeError> {
    let context = runmat_runtime::context::legacy::active().ok_or_else(|| {
        workspace_unavailable("retained workspace write has no active runtime context")
    })?;
    if let Some(service) = context.service_ports().workspace() {
        return service.assign_executor_slot_lease_with_context(&context, index, name, value);
    }
    let frame = target_frame_id(current_owner(), WorkspaceTarget::Current)
        .ok_or_else(|| workspace_unavailable("workspace frame unavailable"))?;
    let prepared = prepare_workspace_frame_assignment_lease_at_slot(
        frame, &context, index, name, value, false,
    )?;
    prepared.checkpoint_before_publication()?;
    prepared.commit();
    Ok(())
}

/// Install one already-owned executor result. Named workspace slots move the
/// Value into an admitted immutable generation before any delegate effect;
/// private compiler slots retain the original zero-copy Vec representation.
pub(crate) fn commit_workspace_slot_value(
    vars: &mut InterpreterVars,
    index: usize,
    compiler_name: Option<&str>,
    value: Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    runmat_runtime::execution::validate_storable_value(&value)?;
    let existing = workspace_slot_name_handle(index);
    let selected_name = compiler_name
        .filter(|name| compiler_workspace_slot_matches(index, name))
        .or_else(|| existing.as_ref().map(WorkspaceName::as_str));
    if let Some(name) = selected_name {
        let context = runmat_runtime::context::legacy::active().ok_or_else(|| {
            workspace_unavailable("workspace slot retention requires an active runtime context")
        })?;
        let retained =
            runmat_runtime::workspace::WorkspaceSlotReadLease::prepare_owned(value, &context)?;
        admit_workspace_slot_lease(index, name, retained)?;
        return Ok(());
    }
    match compiler_name {
        Some(name) => admit_workspace_slot_write_with_name(index, name, &value)?,
        None => admit_workspace_slot_write(index, &value)?,
    }
    if vars.len() <= index {
        vars.resize_owned(index + 1, Value::Num(0.0));
    }
    let retired = vars
        .replace_owned(index, value)
        .ok_or_else(|| workspace_unavailable("executor slot lies outside its frame"))?;
    drop(retired);
    Ok(())
}

/// A function local is normally private to its call frame. Only a slot with
/// an actual declared global binding may publish through the workspace
/// service before its local register changes. Keep the compiler's workspace
/// name (which can differ from the bound session-global name) so the service
/// resolves the typed binding and its original slot.
pub(crate) fn admit_declared_global_slot_write(
    index: usize,
    value: &Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    let name = with_stack(
        current_owner(),
        |stack| -> Result<Option<WorkspaceName>, runmat_runtime::RuntimeError> {
            let Some(frame) = stack.last() else {
                return Ok(None);
            };
            if !frame.global_aliases.has_binding_for_slot(index) {
                return Ok(None);
            }
            let lifecycle = frame.state.slot_lifecycle.get(&index).ok_or_else(|| {
                workspace_unavailable("declared global has no workspace slot identity")
            })?;
            let name = match lifecycle {
                SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name.clone(),
            };
            Ok(Some(name))
        },
    )?;
    if let Some(name) = name {
        admit_named_workspace_write(index, name.as_str(), value)?;
    }
    Ok(())
}

/// Use a compiler-owned name when it identifies the active slot or introduces
/// an otherwise unbound slot. The borrow comes from immutable bytecode, not
/// the workspace's RefCell, so a publication callback may safely re-enter the
/// workspace service.
pub(crate) fn admit_workspace_slot_write_with_name(
    index: usize,
    name: &str,
    value: &Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    let matches = with_stack(current_owner(), |stack| {
        stack.last().is_some_and(|frame| {
            let ws = &frame.state;
            ws.names.get(name).is_none_or(|slot| *slot == index)
                && ws
                    .slot_lifecycle
                    .get(&index)
                    .is_none_or(|slot| slot.name() == name)
        })
    });
    if matches {
        admit_named_workspace_write(index, name, value)
    } else {
        admit_workspace_slot_write(index, value)
    }
}

fn admit_named_workspace_write(
    index: usize,
    name: &str,
    value: &Value,
) -> Result<(), runmat_runtime::RuntimeError> {
    if let Some(context) = runmat_runtime::context::legacy::active() {
        if let Some(service) = context.service_ports().workspace() {
            return service.assign_executor_slot_ref(&context, index, name, value);
        }
    }
    let frame = target_frame_id(current_owner(), WorkspaceTarget::Current)
        .ok_or_else(|| workspace_unavailable("workspace frame unavailable"))?;
    let context = runmat_runtime::context::legacy::active();
    let prepared = prepare_workspace_frame_assignment(frame, context.as_ref(), name, value, false)?;
    prepared.checkpoint_before_publication()?;
    prepared.commit();
    Ok(())
}

/// Read the authoritative language-visible value for a named workspace slot.
/// Compiler temporaries and procedure locals are intentionally absent.
pub fn workspace_slot_value(index: usize) -> Option<Value> {
    with_stack(current_owner(), |stack| {
        let frame = stack.last()?;
        let lifecycle = frame.state.slot_lifecycle.get(&index)?;
        lifecycle
            .is_assigned()
            .then(|| frame.state.values.get(index).map(|value| value.clone()))
            .flatten()
    })
}

/// Snapshot the slots currently assigned in the innermost VM workspace.
/// Internal placeholder values are deliberately excluded.
pub(crate) fn current_workspace_assigned_slots() -> HashSet<usize> {
    with_stack(current_owner(), |stack| {
        stack
            .last()
            .map(|frame| {
                frame
                    .state
                    .slot_lifecycle
                    .iter()
                    .filter_map(|(slot, lifecycle)| lifecycle.is_assigned().then_some(*slot))
                    .collect()
            })
            .unwrap_or_default()
    })
}

pub(crate) fn reset_thread_state_for_tests() {
    activation::reset_for_tests();
    WORKSPACE_STACKS.with(|stacks| stacks.borrow_mut().clear());
    completion::reset_for_tests();
    PENDING_SERVICE_FRAMES.with(|frames| frames.borrow_mut().clear());
    NEXT_WORKSPACE_FRAME_ID.with(|next| next.set(1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake};

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    fn take_report_after(f: impl FnOnce(&mut InterpreterVars)) -> WorkspaceAssignedReport {
        let _ = take_updated_workspace_assigned_report();
        let _ = take_updated_workspace_state();

        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(Vec::new());
        {
            let _guard = set_workspace_state(
                HashMap::new(),
                HashSet::new(),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            f(&mut vars);
        }

        take_updated_workspace_assigned_report().expect("workspace report should be recorded")
    }

    #[test]
    fn transactional_frames_capture_completion_without_live_publication() {
        let _ = take_updated_workspace_state();
        let _ = take_updated_workspace_assigned_report();
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(1.0)]);
        {
            let _guard = install_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut vars,
                WorkspaceFramePolicy::Transactional,
                BaseTargetAuthority::Frame,
                FrameGlobalBindings::default(),
            );
            let frame = with_stack(current_owner(), |stack| stack.last().unwrap().id);
            assert_eq!(workspace_frame_publishes(frame), Some(false));
            workspace_frame_assign_from_executor(frame, "x", Value::Num(7.0)).unwrap();
        }

        let completed = take_updated_workspace_state()
            .expect("transactional caller must receive the completed frame");
        assert_eq!(
            completed.vars.clone_values_for_test(),
            vec![Value::Num(7.0)]
        );
        assert_eq!(completed.names.get("x"), Some(&0));
        assert!(completed.assigned.contains("x"));
    }

    #[test]
    fn detached_source_base_selection_is_explicit_admission_provenance() {
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(Vec::new());
        {
            let _guard = set_workspace_state_with_policy(
                HashMap::new(),
                HashSet::new(),
                &mut vars,
                WorkspaceFramePolicy::TopLevel,
                BaseTargetAuthority::CanonicalService,
                FrameGlobalBindings::default(),
            );
            assert_eq!(
                base_target_authority(),
                Some(BaseTargetAuthority::CanonicalService)
            );
        }
        {
            let _guard = set_workspace_state(
                HashMap::new(),
                HashSet::new(),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            assert_eq!(base_target_authority(), Some(BaseTargetAuthority::Frame));
        }
    }

    #[test]
    fn transient_frames_do_not_become_host_completions() {
        let _ = take_updated_workspace_state();
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(1.0)]);
        {
            let _guard = install_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut vars,
                WorkspaceFramePolicy::Transient,
                BaseTargetAuthority::Frame,
                FrameGlobalBindings::default(),
            );
        }
        assert!(take_updated_workspace_state().is_none());
    }

    #[test]
    fn interleaved_runtime_contexts_keep_workspace_frames_isolated() {
        async fn session(label: &'static str) {
            let mut vars =
                crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::String(
                    label.into(),
                )]);
            let names = HashMap::from([(String::from("value"), 0)]);
            let assigned = HashSet::from([String::from("value")]);
            let _guard = set_transient_workspace_state(
                names,
                assigned,
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(workspace_lookup("value"), Some(Value::String(label.into())));
        }

        fn poll(future: Pin<&mut impl Future<Output = ()>>) -> Poll<()> {
            let waker = std::task::Waker::from(Arc::new(NoopWake));
            future.poll(&mut Context::from_waker(&waker))
        }

        let first = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let second = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        assert_ne!(first.local_identity(), second.local_identity());
        let first_frame = reserve_workspace_service_frame(first.local_identity(), None);
        let second_frame = reserve_workspace_service_frame(second.local_identity(), None);
        let mut first = Box::pin(
            frame_activation(first.local_identity(), first_frame.id)
                .scope(first.scope(session("first"))),
        );
        let mut second = Box::pin(
            frame_activation(second.local_identity(), second_frame.id)
                .scope(second.scope(session("second"))),
        );
        assert!(poll(first.as_mut()).is_pending());
        assert!(poll(second.as_mut()).is_pending());
        assert!(poll(first.as_mut()).is_ready());
        assert!(poll(second.as_mut()).is_ready());
    }

    #[test]
    fn active_frame_never_consumes_a_different_sole_reservation() {
        reset_thread_state_for_tests();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let owner = runtime.local_identity();
        let missing = reserve_workspace_service_frame(owner, None);
        let remaining = reserve_workspace_service_frame(owner, None);
        release_pending_workspace_service_frame(owner, missing.id);
        let activation = frame_activation(owner, missing.id);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            futures::executor::block_on(activation.scope(runtime.scope(async {
                let mut vars =
                    crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(1.0)]);
                let _guard = set_transient_workspace_state(
                    HashMap::from([("x".to_string(), 0)]),
                    HashSet::from(["x".to_string()]),
                    &mut vars,
                    FrameGlobalBindings::default(),
                );
            })));
        }));
        assert!(result.is_err());
        PENDING_SERVICE_FRAMES.with(|frames| {
            assert_eq!(
                frames
                    .borrow()
                    .get(&WorkspaceOwner::Runtime(owner))
                    .and_then(|pending| pending.first())
                    .map(|frame| frame.id),
                Some(remaining.id)
            );
        });
        reset_thread_state_for_tests();
    }

    #[test]
    fn remove_preserves_assignment_report_and_records_removal() {
        let report = take_report_after(|vars| {
            set_workspace_variable("x", Value::Num(1.0), vars).unwrap();
            workspace_remove("x").unwrap();
        });

        assert!(report.names.contains("x"));
        assert!(report.ids.contains(&0));
        assert!(report.removed_names.contains("x"));
        assert!(report.removed_ids.contains(&0));
    }

    #[test]
    fn clear_preserves_assignment_report_and_records_removal() {
        let report = take_report_after(|vars| {
            set_workspace_variable("x", Value::Num(1.0), vars).unwrap();
            workspace_clear().unwrap();
        });

        assert!(report.names.contains("x"));
        assert!(report.ids.contains(&0));
        assert!(report.removed_names.contains("x"));
        assert!(report.removed_ids.contains(&0));
    }

    #[test]
    fn clear_preserves_active_frame_slots() {
        let _ = take_updated_workspace_state();
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![
            Value::Num(11.0),
            Value::Num(0.0),
            Value::Num(0.0),
        ]);
        let mut names = HashMap::new();
        names.insert("x".to_string(), 0);
        let assigned = HashSet::from(["x".to_string()]);

        {
            let _guard =
                set_workspace_state(names, assigned, &mut vars, FrameGlobalBindings::default());
            workspace_clear().unwrap();

            assert_eq!(vars.len(), 3);
            assert_eq!(*vars.get(0).unwrap(), Value::Num(0.0));
            assert_eq!(*vars.get(1).unwrap(), Value::Num(0.0));
            assert_eq!(*vars.get(2).unwrap(), Value::Num(0.0));
            assert_eq!(workspace_lookup("x"), None);
            assert_eq!(workspace_slot_value(0), None);
            assert_eq!(workspace_slot_assigned(0), Some(false));
        }

        let snapshot = take_updated_workspace_state().expect("workspace state should be recorded");
        assert_eq!(snapshot.vars.len(), 3);
        assert!(snapshot.names.is_empty());
        assert!(snapshot.assigned.is_empty());
    }

    #[test]
    fn service_mutations_share_the_canonical_register_control() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runtime_id = runtime.local_identity();
        let reserved = reserve_workspace_service_frame(runtime_id, None);
        let activation = frame_activation(runtime_id, reserved.id);
        futures::executor::block_on(activation.scope(runtime.scope(async {
            let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![
                Value::Num(1.0),
                Value::Num(2.0),
            ]);
            let guard = set_workspace_state(
                HashMap::from([("x".to_string(), 0), ("y".to_string(), 1)]),
                HashSet::from(["x".to_string(), "y".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let frame = with_stack(current_owner(), |stack| stack.last().unwrap().id);

            workspace_frame_assign(frame, "x", Value::Num(9.0)).unwrap();
            assert_eq!(vars, vec![Value::Num(9.0), Value::Num(2.0)]);
            reconcile_workspace_vars(&mut vars);
            assert_eq!(vars, vec![Value::Num(9.0), Value::Num(2.0)]);

            workspace_frame_clear(frame).unwrap();
            reconcile_workspace_vars(&mut vars);
            assert_eq!(vars, vec![Value::Num(0.0), Value::Num(0.0)]);
            drop(guard);
        })));
    }

    #[test]
    fn executor_admission_does_not_schedule_duplicate_register_reconciliation() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runtime_id = runtime.local_identity();
        let reserved = reserve_workspace_service_frame(runtime_id, None);
        let activation = frame_activation(runtime_id, reserved.id);
        futures::executor::block_on(activation.scope(runtime.scope(async {
            let mut vars =
                crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(1.0)]);
            let guard = set_workspace_state(
                HashMap::from([("x".to_string(), 0)]),
                HashSet::from(["x".to_string()]),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let frame = with_stack(current_owner(), |stack| stack.last().unwrap().id);
            workspace_frame_assign_from_executor(frame, "x", Value::Num(7.0)).unwrap();
            let _ = vars.replace_owned(0, Value::Num(7.0));
            reconcile_workspace_vars(&mut vars);
            assert_eq!(vars, vec![Value::Num(7.0)]);
            drop(guard);
        })));
    }

    #[test]
    fn assignment_after_clear_clears_final_removal_marker() {
        let report = take_report_after(|vars| {
            set_workspace_variable("x", Value::Num(1.0), vars).unwrap();
            workspace_clear().unwrap();
            set_workspace_variable("x", Value::Num(2.0), vars).unwrap();
        });

        assert!(report.names.contains("x"));
        assert!(report.removed_names.is_empty());
        assert!(report.removed_ids.is_empty());
    }

    #[test]
    fn assignment_after_remove_reuses_previous_slot() {
        let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(Vec::new());
        let _ = take_updated_workspace_state();
        {
            let _guard = set_workspace_state(
                HashMap::new(),
                HashSet::new(),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            set_workspace_variable("x", Value::Num(1.0), &mut vars).unwrap();
            set_workspace_variable("z", Value::Num(9.0), &mut vars).unwrap();
            workspace_remove("x").unwrap();
            set_workspace_variable("x", Value::Num(42.0), &mut vars).unwrap();
            assert_eq!(workspace_lookup("x"), Some(Value::Num(42.0)));
            assert_eq!(*vars.get(0).unwrap(), Value::Num(42.0));
        }

        let snapshot = take_updated_workspace_state().expect("workspace state should be recorded");
        assert_eq!(snapshot.names.get("x"), Some(&0));
        assert_eq!(snapshot.names.get("z"), Some(&1));
        assert!(snapshot.assigned.contains("x"));
        assert!(snapshot.assigned.contains("z"));
        assert_eq!(*snapshot.vars.get(0).unwrap(), Value::Num(42.0));
    }

    #[test]
    fn dynamic_assignment_publishes_active_global_but_clear_only_unbinds_frame() {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (runtime, activation) =
            super::super::workspace_service::VmWorkspaceService::install(runtime, None);
        futures::executor::block_on(activation.scope(runtime.scope(async {
            let aliases = FrameGlobalBindings::default();
            aliases.declare_named(0, "alpha".into());
            runmat_runtime::workspace::session::store_global_named("alpha", Value::Num(7.0))
                .expect("seed named global");
            let mut vars =
                crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(7.0)]);
            let _guard = set_workspace_state(
                HashMap::from([("alpha".into(), 0)]),
                HashSet::from(["alpha".into()]),
                &mut vars,
                aliases.clone(),
            );
            workspace_assign("alpha", Value::Num(8.0)).unwrap();
            assert_eq!(
                runmat_runtime::workspace::session::global_value("alpha"),
                Some(Value::Num(8.0))
            );
            workspace_remove("alpha").unwrap();
            assert_eq!(aliases.name_for_slot(0), None);
            workspace_assign("alpha", Value::Num(9.0)).unwrap();
            assert_eq!(
                runmat_runtime::workspace::session::global_value("alpha"),
                Some(Value::Num(8.0)),
                "ordinary clear removes the frame binding, not the global value"
            );
            assert_eq!(workspace_lookup("alpha"), Some(Value::Num(9.0)));
            assert_eq!(*vars.get(0).unwrap(), Value::Num(9.0));
            reconcile_workspace_vars(&mut vars);
            assert_eq!(*vars.get(0).unwrap(), Value::Num(9.0));
        })));
    }

    #[test]
    fn workspace_frames_do_not_retain_raw_execution_storage_pointers() {
        let source = include_str!("workspace.rs");
        let raw_vec_pointer = ["*mut", "Vec<Value>"].join(" ");
        let raw_value_pointer = ["*const", "Value"].join(" ");
        assert!(!source.contains(&raw_vec_pointer));
        assert!(!source.contains(&raw_value_pointer));
    }

    #[test]
    fn asynchronous_language_visible_writers_use_checked_slot_admission() {
        let writers = [
            ("fusion", include_str!("../accel/fusion.rs")),
            (
                "prepared outputs",
                include_str!("../interpreter/dispatch/prepared_outputs/commit.rs"),
            ),
            (
                "parfor",
                include_str!("../interpreter/dispatch/parallel/parfor.rs"),
            ),
            (
                "spmd",
                include_str!("../interpreter/dispatch/parallel/spmd.rs"),
            ),
            (
                "expanded nested calls",
                include_str!("../interpreter/dispatch/calls/dispatch/expanded.rs"),
            ),
            (
                "nested user calls",
                include_str!("../interpreter/dispatch/calls/dispatch/user/nested.rs"),
            ),
            (
                "catch variables",
                include_str!("../interpreter/dispatch/exceptions.rs"),
            ),
        ];
        for (domain, source) in writers {
            assert!(
                source.contains("commit_workspace_slot_value"),
                "{domain} bypasses checked language-visible slot admission"
            );
        }
    }
}

#[cfg(test)]
#[path = "workspace/lexical_tests.rs"]
mod lexical_tests;
