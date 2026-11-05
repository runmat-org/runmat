use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::{Rc, Weak},
};

use runmat_execution::resource::ExactShared;
use runmat_mir::{MirLocalId, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeWorkspaceService;
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
use runmat_runtime::workspace::{
    CapturedWorkspaceReadView, DetachedWorkspaceValues, GlobalBindingTarget,
    LogicalWorkspaceGeneration, WorkspaceDetachSelection, WorkspaceRevision,
    WorkspaceSlotReadLease,
};
use runmat_types::BindingId;
use runmat_types::{EnvironmentEffect, WorkspaceEffect};
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError, Value,
};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

mod operation;
mod names;
mod capture;
use names::NativeWorkspaceNames;

/// One assigned interactive binding supplied to a native script invocation.
///
/// Binding identity selects the exact Native IR local. The retained name is
/// the MATLAB workspace identity used by dynamic workspace builtins.
/// The value is retained by `NativeWorkspaceInput`'s logical generation;
/// metadata must not duplicate payloads or compare language values to
/// validate caller identity.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeWorkspaceBinding {
    pub binding: BindingId,
    pub name: String,
}

/// One source-visible workspace name and its exact session-global storage
/// destination. The target is independent of both the source spelling and the
/// Native IR local slot.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeWorkspaceGlobalBinding {
    pub binding: BindingId,
    pub name: String,
    pub target: GlobalBindingTarget,
}

/// Transactional interactive workspace input for one native invocation.
#[derive(Clone, Debug)]
pub struct NativeWorkspaceInput {
    /// Exact immutable caller workspace, including dynamic names not named by
    /// this source. The generation owns its Value and managed-handle roots.
    generation: LogicalWorkspaceGeneration,
    /// Function-scoped semantic names for every caller binding retained by
    /// the synthetic entrypoint, including bindings initially unassigned.
    pub local_names: BTreeMap<BindingId, String>,
    /// Exact semantic locals that may read or write caller bindings.
    pub bindings: Vec<NativeWorkspaceBinding>,
    /// Semantic caller bindings that resolve through the session-global store.
    /// Values remain owned by that store; this list carries alias identity only.
    pub global_bindings: Vec<NativeWorkspaceGlobalBinding>,
}

impl NativeWorkspaceInput {
    /// A standalone script begins with no caller workspace. Construct its
    /// source through the same exact generation owner as populated inputs.
    pub fn empty_for_context(
        context: &runmat_runtime::context::RuntimeContext,
        local_names: BTreeMap<BindingId, String>,
    ) -> Result<Self, Box<runmat_runtime::RuntimeError>> {
        let base = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let generation = LogicalWorkspaceGeneration::capture(
            context,
            &base,
            &context.session_variable_authority(),
        )
        .map_err(Box::new)?;
        Ok(Self::from_generation(
            generation,
            local_names,
            Vec::new(),
            Vec::new(),
        ))
    }

    pub fn from_generation(
        generation: LogicalWorkspaceGeneration,
        local_names: BTreeMap<BindingId, String>,
        bindings: Vec<NativeWorkspaceBinding>,
        global_bindings: Vec<NativeWorkspaceGlobalBinding>,
    ) -> Self {
        Self {
            generation,
            local_names,
            bindings,
            global_bindings,
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.generation.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.generation.iter()
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        self.generation.global_target(name)
    }

    pub fn profile_values(&self) -> impl Iterator<Item = &Value> {
        self.generation.iter().map(|(_, value)| value)
    }

    #[cfg(test)]
    pub(super) fn from_test_values(
        values: BTreeMap<String, Value>,
        local_names: BTreeMap<BindingId, String>,
        bindings: Vec<NativeWorkspaceBinding>,
        global_bindings: Vec<NativeWorkspaceGlobalBinding>,
    ) -> Self {
        let context = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let base = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
        let session = context.session_variable_authority();
        for (name, value) in values {
            match global_bindings.iter().find(|binding| binding.name == name) {
                Some(binding) => match &binding.target {
                    GlobalBindingTarget::Named(target) => session
                        .store_global_with_context(&context, target.as_str(), value)
                        .expect("valid fixture named global"),
                    GlobalBindingTarget::LegacySlot(slot) => session
                        .store_global_slot_with_context(&context, *slot, value)
                        .expect("valid fixture legacy global"),
                },
                None => delta
                    .assign(name, value)
                    .expect("valid fixture workspace value"),
            }
        }
        for binding in &global_bindings {
            delta.bind_global_target(binding.name.clone(), binding.target.clone());
        }
        base.apply(delta).expect("install fixture workspace values");
        let generation = LogicalWorkspaceGeneration::capture(&context, &base, &session)
            .expect("capture fixture workspace generation");
        Self::from_generation(generation, local_names, bindings, global_bindings)
    }

    pub(super) fn frame_global_bindings(
        &self,
        function: &runmat_native_codegen::NativeFunction,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<FrameGlobalBindings> {
        let aliases = FrameGlobalBindings::default();
        let mut seen = BTreeSet::new();
        for binding in &self.global_bindings {
            if !seen.insert(binding.binding) {
                return Err(NativeExecutorError::Host(
                    "native workspace input contains a duplicate global binding".into(),
                ));
            }
            let local = function
                .locals
                .iter()
                .find(|local| local.binding == Some(binding.binding))
                .ok_or_else(|| {
                    NativeExecutorError::Host(format!(
                        "native global binding '{}' is unavailable",
                        binding.name
                    ))
                })?;
            if local.kind != runmat_native_codegen::NativeLocalKind::Binding
                || self.local_names.get(&binding.binding) != Some(&binding.name)
            {
                return Err(NativeExecutorError::Host(format!(
                    "native global binding '{}' does not match Native IR",
                    binding.name
                )));
            }
            let prepared = aliases
                .prepare_target(context, local.id.0 as usize, &binding.target)
                .map_err(|cause| {
                    NativeExecutorError::from(super::global_target::alias_admission_error(
                        context, cause,
                    ))
                })?;
            prepared.checkpoint().map_err(|cause| {
                NativeExecutorError::from(super::global_target::alias_admission_error(
                    context, cause,
                ))
            })?;
            prepared.commit();
        }
        Ok(aliases)
    }
}

/// Retained logical workspace generation after a successful invocation. It
/// shares the Runtime projection/session owners and never copies Values into
/// a terminal Native map.
#[derive(Clone, Debug)]
pub struct NativeWorkspaceSnapshot {
    owner: ExactShared<NativeWorkspaceSnapshotPayload>,
}

#[derive(Debug)]
struct NativeWorkspaceSnapshotPayload {
    view: CapturedWorkspaceReadView,
    // ExactShared releases its header before dropping this payload. The
    // immutable generation and lexical leases retire before the claim.
    _claim: Option<ExactHostAllocationClaim>,
}

impl PartialEq for NativeWorkspaceSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
            && self
                .owner
                .get()
                .view
                .global_bindings()
                .eq(other.owner.get().view.global_bindings())
    }
}

impl NativeWorkspaceSnapshot {
    fn prepare(
        context: &runmat_runtime::context::RuntimeContext,
        view: CapturedWorkspaceReadView,
    ) -> NativeExecutorResult<Self> {
        let bytes = ExactShared::<NativeWorkspaceSnapshotPayload>::requested_bytes();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes)
                .map_err(|_| snapshot_admission_error(context, HostAllocationError::Overflow))?,
            1,
        )
        .map_err(|cause| snapshot_admission_error(context, cause))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|cause| snapshot_admission_error(context, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| snapshot_admission_error(context, cause))?;
        let build = |permit: Option<&dyn ExactHostAllocationPermit>| {
            let claim = permit
                .map(|permit| permit.reserve_exact_claim(bytes))
                .transpose()?;
            ExactShared::try_new(NativeWorkspaceSnapshotPayload {
                view,
                _claim: claim,
            })
            .map(|owner| Self { owner })
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
        };
        match receipt {
            Some(receipt) => receipt
                .with_output_allocation(context, |permit| build(Some(permit)))
                .map_err(|cause| snapshot_admission_error(context, cause)),
            None => build(None).map_err(|cause| {
                snapshot_admission_error(context, HostAllocationError::Numeric(cause))
            }),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.owner.get().view.iter()
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.owner.get().view.get(name).is_some()
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.owner.get().view.get(name)
    }

    pub fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        self.owner.get().view.global_target(name)
    }
}

fn snapshot_admission_error(
    context: &runmat_runtime::context::RuntimeContext,
    cause: HostAllocationError,
) -> NativeExecutorError {
    NativeExecutorError::from(crate::error::allocation_error(
        Some(context),
        format!("native workspace snapshot admission failed: {cause}"),
        cause,
    ))
}

struct NativeWorkspaceFrame {
    runtime: Option<runmat_runtime::context::RuntimeContextLocalId>,
    lexical: RefCell<Option<NativeLexicalWorkspaceBindings>>,
    externally_mutated: Cell<bool>,
    active: Cell<bool>,
}

#[derive(Clone)]
struct NativeLexicalWorkspaceBindings {
    environment: runmat_value::LexicalEnvironment,
    by_name: BTreeMap<String, runmat_native_codegen::NativeLexicalBinding>,
}

fn private_workspace_service(
    generation: Option<&runmat_runtime::workspace::LogicalWorkspaceGeneration>,
    context: Option<&runmat_runtime::context::RuntimeContext>,
) -> NativeExecutorResult<(
    Rc<dyn RuntimeWorkspaceService>,
    runmat_runtime::workspace::BaseWorkspaceOwner,
)> {
    let (owner, service) = match generation {
        Some(generation) => {
            runmat_runtime::workspace::BaseWorkspaceProjectionService::private_from_generation(
                generation, context,
            )
        }
        None => runmat_runtime::workspace::BaseWorkspaceProjectionService::private_empty(context),
    }
    .map_err(NativeExecutorError::from)?;
    Ok((Rc::new(service), owner))
}

#[derive(Clone, Copy)]
enum NativeWorkspaceMutationOrigin {
    ExecutorStore,
    CanonicalRefresh,
}

pub(super) struct NativeWorkspaceState {
    frame: Rc<NativeWorkspaceFrame>,
    observed_revision: Cell<Option<WorkspaceRevision>>,
    local_names: NativeWorkspaceNames,
    publish: bool,
    global_aliases: FrameGlobalBindings,
    canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
    caller: Option<Rc<dyn RuntimeWorkspaceService>>,
    _private_owner: Option<runmat_runtime::workspace::BaseWorkspaceOwner>,
    session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
}

impl NativeWorkspaceState {
    /// Unbound fixture construction; production frames use `new_for_context`.
    #[cfg(test)]
    pub(super) fn new(
        function: &runmat_native_codegen::NativeFunction,
        input: NativeWorkspaceInput,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
    ) -> NativeExecutorResult<Self> {
        Self::new_with_runtime(
            function,
            input,
            global_aliases,
            canonical,
            session_variables,
            None,
        )
    }

    pub(super) fn new_for_context(
        function: &runmat_native_codegen::NativeFunction,
        input: NativeWorkspaceInput,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<Self> {
        Self::new_with_runtime(
            function,
            input,
            global_aliases,
            canonical,
            session_variables,
            Some(context),
        )
    }

    fn new_with_runtime(
        function: &runmat_native_codegen::NativeFunction,
        input: NativeWorkspaceInput,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
        context: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> NativeExecutorResult<Self> {
        let input_revision = input.generation.base_revision();
        let local_names = NativeWorkspaceNames::prepare(function, Some(&input), context)?;
        for (slot, name) in local_names.iter() {
            if let Some(target) = global_aliases.binding_handle_for_slot(slot) {
                if input.generation.global_target(name) != Some(&target) {
                    return Err(NativeExecutorError::Host(format!(
                        "native global binding '{}' does not match the caller generation target",
                        name
                    )));
                }
            }
        }
        let mut seen_bindings = BTreeSet::new();
        for binding in &input.global_bindings {
            if input.generation.global_target(&binding.name) != Some(&binding.target) {
                return Err(NativeExecutorError::Host(format!(
                    "native global binding '{}' does not match the caller generation target",
                    binding.name
                )));
            }
        }
        for supplied in input.bindings {
            if !seen_bindings.insert(supplied.binding) {
                return Err(NativeExecutorError::Host(
                    "native workspace input contains a duplicate binding".into(),
                ));
            }
            let local = function
                .locals
                .iter()
                .find(|local| local.binding == Some(supplied.binding))
                .ok_or_else(|| {
                    NativeExecutorError::Host(format!(
                        "native workspace binding '{}' is unavailable",
                        supplied.name
                    ))
                })?;
            if local.kind != runmat_native_codegen::NativeLocalKind::Binding
                || input.local_names.get(&supplied.binding) != Some(&supplied.name)
            {
                return Err(NativeExecutorError::Host(format!(
                    "native workspace binding '{}' does not match Native IR",
                    supplied.name
                )));
            }
            if input.generation.get(&supplied.name).is_none() {
                return Err(NativeExecutorError::Host(format!(
                    "native workspace binding '{}' is missing from the caller snapshot",
                    supplied.name
                )));
            }
        }
        let caller = canonical.clone();
        let (canonical, private_owner) = match canonical {
            Some(canonical) => (Some(canonical), None),
            None => {
                let (service, owner) = private_workspace_service(Some(&input.generation), context)?;
                (Some(service), Some(owner))
            }
        };
        Ok(Self {
            frame: Rc::new(NativeWorkspaceFrame {
                runtime: context.map(runmat_runtime::context::RuntimeContext::local_identity),
                lexical: RefCell::new(None),
                externally_mutated: Cell::new(false),
                active: Cell::new(true),
            }),
            observed_revision: Cell::new(Some(input_revision)),
            local_names,
            publish: true,
            global_aliases,
            canonical,
            caller,
            _private_owner: private_owner,
            session_variables,
        })
    }

    /// Unbound fixture construction; production frames use `function_frame_for_context`.
    #[cfg(test)]
    pub(super) fn function_frame(
        function: &runmat_native_codegen::NativeFunction,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
    ) -> Self {
        Self::function_frame_with_runtime(
            function,
            global_aliases,
            canonical,
            session_variables,
            None,
        )
        .expect("unbound fixture workspace frame")
    }

    pub(super) fn function_frame_for_context(
        function: &runmat_native_codegen::NativeFunction,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<Self> {
        Self::function_frame_with_runtime(
            function,
            global_aliases,
            canonical,
            session_variables,
            Some(context),
        )
    }

    fn function_frame_with_runtime(
        function: &runmat_native_codegen::NativeFunction,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
        context: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> NativeExecutorResult<Self> {
        let local_names = NativeWorkspaceNames::prepare(function, None, context)?;
        let (service, owner) = private_workspace_service(None, context)?;
        let (current, private_owner) = (Some(service), Some(owner));
        if let (Some(context), Some(current)) = (context, current.as_ref()) {
            for (slot, name) in local_names.iter() {
                if let Some(target) = global_aliases.binding_handle_for_slot(slot) {
                    current.bind_base_global_target_with_context(context, name, &target)?;
                }
            }
        }
        Ok(Self {
            frame: Rc::new(NativeWorkspaceFrame {
                runtime: context.map(runmat_runtime::context::RuntimeContext::local_identity),
                lexical: RefCell::new(None),
                externally_mutated: Cell::new(false),
                active: Cell::new(true),
            }),
            observed_revision: Cell::new(None),
            local_names,
            publish: false,
            global_aliases,
            canonical: current,
            caller: canonical,
            _private_owner: private_owner,
            session_variables,
        })
    }

    fn service_value(&self) -> NativeWorkspaceService {
        NativeWorkspaceService {
            frame: Rc::downgrade(&self.frame),
            global_aliases: self.global_aliases.clone(),
            local_names: self.local_names.clone(),
            canonical: self.canonical.as_ref().map(Rc::downgrade),
            caller: self.caller.as_ref().map(Rc::downgrade),
            private_current: self._private_owner.is_some(),
            publish: self.publish,
            session_variables: self.session_variables.clone(),
        }
    }

    pub(super) fn service(&self) -> Rc<dyn RuntimeWorkspaceService> {
        Rc::new(self.service_value())
    }

    pub(super) fn attach_lexical_bindings(
        &self,
        function: &runmat_native_codegen::NativeFunction,
        environment: Option<&runmat_value::LexicalEnvironment>,
    ) -> NativeExecutorResult<()> {
        if function.lexical_bindings.is_empty() {
            return Ok(());
        }
        let environment = environment.ok_or_else(|| {
            NativeExecutorError::Host("native lexical workspace has no activation".into())
        })?;
        let mut by_name = BTreeMap::new();
        for binding in &function.lexical_bindings {
            let name = self
                .local_names
                .name(binding.local.0 as usize)
                .ok_or_else(|| {
                    NativeExecutorError::Host("native lexical binding has no workspace name".into())
                })?;
            if by_name.insert(name.to_owned(), *binding).is_some() {
                return Err(NativeExecutorError::Host(
                    "native lexical workspace has duplicate binding names".into(),
                ));
            }
        }
        let mut slot = self.frame.lexical.try_borrow_mut().map_err(|_| {
            NativeExecutorError::Host("native lexical workspace is borrowed".into())
        })?;
        if slot.is_some() {
            return Err(NativeExecutorError::Host(
                "native lexical workspace was installed twice".into(),
            ));
        }
        *slot = Some(NativeLexicalWorkspaceBindings {
            environment: environment.clone(),
            by_name,
        });
        Ok(())
    }

    pub(super) fn needs_reconciliation(&self) -> bool {
        self.frame.externally_mutated.get()
    }

    pub(super) fn needs_reconciliation_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<bool> {
        if self.needs_reconciliation() {
            return Ok(true);
        }
        let revision = self.current_revision_with_context(context)?;
        Ok(revision != self.observed_revision.get())
    }

    pub(super) fn current_revision_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<Option<WorkspaceRevision>> {
        operation::validate_frame_context(&self.frame, Some(context))?;
        self.canonical
            .as_ref()
            .ok_or_else(|| NativeExecutorError::from(native_workspace_closed()))?
            .workspace_revision_with_context(context)
            .map_err(NativeExecutorError::from)
    }

    pub(super) fn record_observed_revision(&self, revision: Option<WorkspaceRevision>) {
        self.observed_revision.set(revision);
    }

    pub(super) fn mark_reconciled(&self) {
        self.frame.externally_mutated.set(false);
    }

    pub(super) fn capture_generation_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<LogicalWorkspaceGeneration> {
        operation::validate_frame_context(&self.frame, Some(context))?;
        if !self.frame.active.get() {
            return Err(NativeExecutorError::from(native_workspace_closed()));
        }
        if !self.publish {
            if let Some(caller) = &self.caller {
                caller.ensure_effect_authority_with_context(context)?;
            }
        }
        self.canonical
            .as_ref()
            .ok_or_else(|| NativeExecutorError::from(native_workspace_closed()))?
            .capture_workspace_generation_with_context(context)?
            .ok_or_else(|| {
                NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                    "RunMat:Workspace:GenerationUnavailable",
                    "native workspace owner cannot retain a live generation",
                ))
            })
    }

    pub(super) fn initial_value_lease(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
    ) -> NativeExecutorResult<Option<WorkspaceSlotReadLease>> {
        self.lookup_local_lease(context, slot)
    }

    fn shared_lexical_binding(&self, slot: usize) -> NativeExecutorResult<bool> {
        let Some(name) = self.local_names.name(slot) else {
            return Ok(false);
        };
        let bindings = self
            .frame
            .lexical
            .try_borrow()
            .map_err(workspace_read_borrow_error)?;
        Ok(bindings
            .as_ref()
            .and_then(|owner| owner.by_name.get(name))
            .is_some_and(|binding| {
                binding.kind != runmat_native_codegen::NativeLexicalBindingKind::Transit
            }))
    }

    pub(super) fn lookup_local_lease(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
    ) -> NativeExecutorResult<Option<WorkspaceSlotReadLease>> {
        if let Some(name) = self.local_names.name(slot) {
            let lexical = {
                let bindings = self
                    .frame
                    .lexical
                    .try_borrow()
                    .map_err(workspace_read_borrow_error)?;
                bindings.as_ref().and_then(|owner| {
                    owner.by_name.get(name).copied().and_then(|binding| {
                        (binding.kind != runmat_native_codegen::NativeLexicalBindingKind::Transit)
                            .then(|| (owner.environment.clone(), binding.binding))
                    })
                })
            };
            if let Some((environment, binding)) = lexical {
                return runmat_runtime::call::lexical::read_binding_lease_with_context(
                    &environment,
                    binding,
                    context,
                )
                .map(|lease| lease.map(WorkspaceSlotReadLease::from))
                .map_err(NativeExecutorError::from);
            }
        }
        let generation = self.capture_generation_with_context(context)?;
        Ok(self
            .local_names
            .name(slot)
            .and_then(|name| generation.read_lease(name))
            .map(WorkspaceSlotReadLease::from))
    }

    pub(super) fn local_slots(&self) -> impl Iterator<Item = (usize, &str)> {
        self.local_names.iter()
    }

    #[cfg(test)]
    pub(super) fn initialize_local(&self, slot: usize, value: Value) -> NativeExecutorResult<()> {
        self.initialize_local_in_context(None, slot, value)
    }

    pub(super) fn initialize_local_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
        value: Value,
    ) -> NativeExecutorResult<()> {
        self.initialize_local_in_context(Some(context), slot, value)
    }

    /// Move a named input into the canonical projection before the arena is
    /// seeded. An unnamed or session-global local remains owned by the caller.
    pub(super) fn initialize_local_owned_or_return(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
        value: Value,
    ) -> NativeExecutorResult<Option<Value>> {
        operation::validate_frame_context(&self.frame, Some(context))?;
        let Some(name) = self.local_names.name(slot) else {
            return Ok(Some(value));
        };
        if self.global_aliases.has_binding_for_slot(slot) {
            return Ok(Some(value));
        }
        if self.shared_lexical_binding(slot)? {
            return Ok(Some(value));
        }
        let current = self
            .canonical
            .as_ref()
            .ok_or_else(|| NativeExecutorError::from(native_workspace_closed()))?;
        current.assign_with_context(context, name, value)?;
        Ok(None)
    }

    fn initialize_local_in_context(
        &self,
        context: Option<&runmat_runtime::context::RuntimeContext>,
        slot: usize,
        value: Value,
    ) -> NativeExecutorResult<()> {
        let Some(name) = self.local_names.name(slot) else {
            return Ok(());
        };
        operation::validate_frame_context(&self.frame, context)?;
        if self.shared_lexical_binding(slot)? {
            return Ok(());
        }
        if !self.global_aliases.has_binding_for_slot(slot) {
            if let Some(current) = &self.canonical {
                match context {
                    Some(context) => current.assign_with_context(context, name, value)?,
                    None => current.assign(name, value)?,
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn synchronize_local(&self, slot: usize, value: Value) -> NativeExecutorResult<()> {
        self.synchronize_local_with_origin(
            None,
            slot,
            value,
            NativeWorkspaceMutationOrigin::ExecutorStore,
        )
    }

    pub(super) fn synchronize_local_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
        value: Value,
    ) -> NativeExecutorResult<()> {
        self.synchronize_local_with_origin(
            Some(context),
            slot,
            value,
            NativeWorkspaceMutationOrigin::ExecutorStore,
        )
    }

    pub(super) fn refresh_local_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
        value: Value,
    ) -> NativeExecutorResult<()> {
        self.synchronize_local_with_origin(
            Some(context),
            slot,
            value,
            NativeWorkspaceMutationOrigin::CanonicalRefresh,
        )
    }

    fn synchronize_local_with_origin(
        &self,
        context: Option<&runmat_runtime::context::RuntimeContext>,
        slot: usize,
        value: Value,
        origin: NativeWorkspaceMutationOrigin,
    ) -> NativeExecutorResult<()> {
        // A private function frame still inherits its caller's revocation.
        // Check that lineage before publishing any frame-local, lexical,
        // session-global, or canonical store; the private current service
        // alone cannot prove that its caller remains executable.
        self.service_value().ensure_mutation_in_operation(context)?;
        if let Some(name) = self.local_names.name(slot) {
            let shared_lexical = {
                let bindings = self
                    .frame
                    .lexical
                    .try_borrow()
                    .map_err(workspace_read_borrow_error)?;
                bindings
                    .as_ref()
                    .and_then(|owner| owner.by_name.get(name))
                    .is_some_and(|binding| {
                        binding.kind != runmat_native_codegen::NativeLexicalBindingKind::Transit
                    })
            };
            if let Some(canonical) = self.canonical.as_ref() {
                if shared_lexical {
                    // HostState writes the lexical cell after this authority
                    // check; no second workspace value is published.
                    if let Some(context) = context {
                        canonical.ensure_mutation_authority_with_context(context)?;
                    } else {
                        canonical.ensure_mutation_authority()?;
                    }
                } else if self._private_owner.is_some()
                    && !self.global_aliases.has_binding_for_slot(slot)
                {
                    // This adapter receives an owned Value, including prepared
                    // varargout/aggregate replacements. Preserve that transfer
                    // instead of asking Base to certify a borrowed alias.
                    if let Some(context) = context {
                        canonical.assign_with_context(context, name, value)?;
                    } else {
                        canonical.assign(name, value)?;
                    }
                } else if self.publish && !self.global_aliases.has_binding_for_slot(slot) {
                    match origin {
                        NativeWorkspaceMutationOrigin::ExecutorStore => {
                            if let Some(context) = context {
                                canonical.assign_executor_slot_ref(context, slot, name, &value)?;
                            } else {
                                canonical.assign_executor_slot(slot, name, value.clone())?;
                            }
                        }
                        NativeWorkspaceMutationOrigin::CanonicalRefresh => {
                            if let Some(context) = context {
                                canonical.assign_ref_with_context(context, name, &value)?;
                            } else {
                                canonical.assign(name, value.clone())?;
                            }
                        }
                    }
                } else {
                    // No canonical publication occurs for a frame-local or
                    // session-global alias. This is the existing mutation-only
                    // checkpoint, not an effect/reconciliation request.
                    if let Some(context) = context {
                        canonical.ensure_mutation_authority_with_context(context)?;
                    } else {
                        canonical.ensure_mutation_authority()?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn snapshot_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<Option<NativeWorkspaceSnapshot>> {
        if !self.publish {
            return Ok(None);
        }
        capture::capture_read_view(&self.service_value(), context)
            .map_err(NativeExecutorError::from)?
            .map(|view| NativeWorkspaceSnapshot::prepare(context, view))
            .transpose()
    }
}

impl Drop for NativeWorkspaceState {
    fn drop(&mut self) {
        self.frame.active.set(false);
    }
}

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "workspace/lexical_tests.rs"]
mod lexical_tests;

#[cfg(test)]
mod gc_tests;

fn workspace_read_borrow_error(error: std::cell::BorrowError) -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "NativeWorkspaceBorrow",
        format!("native workspace read is reentrant: {error}"),
    )
}

fn missing_session_variable_authority() -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:SessionAuthorityMissing",
        "native workspace service has no session-variable authority",
    )
}

struct NativeWorkspaceService {
    frame: Weak<NativeWorkspaceFrame>,
    global_aliases: FrameGlobalBindings,
    local_names: NativeWorkspaceNames,
    canonical: Option<Weak<dyn RuntimeWorkspaceService>>,
    caller: Option<Weak<dyn RuntimeWorkspaceService>>,
    private_current: bool,
    publish: bool,
    session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
}

struct NativeWorkspaceClosed;

impl From<NativeWorkspaceClosed> for runmat_runtime::RuntimeError {
    fn from(_: NativeWorkspaceClosed) -> Self {
        native_workspace_closed()
    }
}

impl From<NativeWorkspaceClosed> for Box<runmat_runtime::RuntimeError> {
    fn from(_: NativeWorkspaceClosed) -> Self {
        Box::new(native_workspace_closed())
    }
}

impl NativeWorkspaceService {
    fn frame(&self) -> Result<Rc<NativeWorkspaceFrame>, NativeWorkspaceClosed> {
        let frame = self.frame.upgrade().ok_or(NativeWorkspaceClosed)?;
        if !frame.active.get() {
            return Err(NativeWorkspaceClosed);
        }
        Ok(frame)
    }

    fn detach_in_operation(
        &self,
        context: Option<&runmat_runtime::context::RuntimeContext>,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, Box<runmat_runtime::RuntimeError>> {
        self.ensure_effect_in_operation(context)?;
        let frame = self.frame()?;
        let canonical = self
            .canonical_for_operation()?
            .ok_or_else(native_workspace_closed)?;
        let detached = match context {
            Some(context) => canonical.detach_selected_with_context(context, selection)?,
            None => canonical.detach_selected(selection)?,
        };
        match selection {
            WorkspaceDetachSelection::All => {
                self.global_aliases.unbind_all();
            }
            WorkspaceDetachSelection::Names(names) => {
                for name in names {
                    self.global_aliases.unbind_name(name);
                }
            }
            WorkspaceDetachSelection::Except(excluded) => {
                for (_, name) in self.local_names.iter() {
                    if !excluded.iter().any(|excluded| excluded == name) {
                        self.global_aliases.unbind_name(name);
                    }
                }
            }
        }
        frame.externally_mutated.set(true);
        Ok(detached)
    }
}

impl RuntimeWorkspaceService for NativeWorkspaceService {
    fn capture_workspace_generation_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Option<LogicalWorkspaceGeneration>, runmat_runtime::RuntimeError> {
        self.ensure_effect_in_operation(Some(context))
            .map_err(|error| *error)?;
        self.canonical
            .as_ref()
            .and_then(Weak::upgrade)
            .ok_or_else(native_workspace_closed)?
            .capture_workspace_generation_with_context(context)
    }

    fn capture_dynamic_workspace_target_with_context(
        self: Rc<Self>,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<
        Option<runmat_runtime::workspace::DynamicWorkspaceTargetCapture>,
        runmat_runtime::RuntimeError,
    > {
        capture::capture_target(self, context).map_err(|error| *error)
    }

    fn caller_workspace_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Option<Rc<dyn RuntimeWorkspaceService>>, runmat_runtime::RuntimeError> {
        self.ensure_effect_in_operation(Some(context))
            .map_err(|error| *error)?;
        Ok(self.caller.as_ref().and_then(Weak::upgrade))
    }

    fn detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, runmat_runtime::RuntimeError> {
        self.detach_in_operation(None, selection)
            .map_err(|error| *error)
    }

    fn detach_selected_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, runmat_runtime::RuntimeError> {
        self.detach_in_operation(Some(context), selection)
            .map_err(|error| *error)
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let frame = self.frame.upgrade()?;
        frame.active.get().then_some(())?;
        self.canonical
            .as_ref()?
            .upgrade()?
            .visit_retained_values(visitor)
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        let frame = self.frame.upgrade()?;
        frame.active.get().then_some(())?;
        frame.runtime.is_none().then_some(())?;
        self.try_lookup_in_operation(None, name).ok().flatten()
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        let Some(frame) = self.frame.upgrade() else {
            return Vec::new();
        };
        if !frame.active.get() {
            return Vec::new();
        }
        if frame.runtime.is_some() {
            return Vec::new();
        }
        self.try_snapshot_in_operation(None).unwrap_or_default()
    }

    fn global_names(&self) -> Vec<String> {
        let Some(frame) = self.frame.upgrade() else {
            return Vec::new();
        };
        if !frame.active.get() {
            return Vec::new();
        }
        if frame.runtime.is_some() {
            return Vec::new();
        }
        match self.session_variables.as_ref() {
            Some(authority) => authority.global_names().unwrap_or_default(),
            #[cfg(test)]
            None => runmat_runtime::workspace::session::global_names(),
            #[cfg(not(test))]
            None => Vec::new(),
        }
    }

    fn try_lookup(&self, name: &str) -> Result<Option<Value>, runmat_runtime::RuntimeError> {
        self.try_lookup_in_operation(None, name)
            .map_err(|error| *error)
    }

    fn try_lookup_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        name: &str,
    ) -> Result<Option<Value>, runmat_runtime::RuntimeError> {
        self.try_lookup_in_operation(Some(context), name)
            .map_err(|error| *error)
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, runmat_runtime::RuntimeError> {
        self.try_snapshot_in_operation(None).map_err(|error| *error)
    }

    fn try_snapshot_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Vec<(String, Value)>, runmat_runtime::RuntimeError> {
        self.try_snapshot_in_operation(Some(context))
            .map_err(|error| *error)
    }

    fn try_global_names(&self) -> Result<Vec<String>, runmat_runtime::RuntimeError> {
        self.try_global_names_in_operation(None)
            .map_err(|error| *error)
    }

    fn try_global_names_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Vec<String>, runmat_runtime::RuntimeError> {
        self.try_global_names_in_operation(Some(context))
            .map_err(|error| *error)
    }

    fn ensure_effect_authority(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_in_operation(None)
            .map_err(|error| *error)
    }

    fn ensure_effect_authority_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_in_operation(Some(context))
            .map_err(|error| *error)
    }

    fn ensure_mutation_authority(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_mutation_in_operation(None)
            .map_err(|error| *error)
    }

    fn ensure_mutation_authority_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_mutation_in_operation(Some(context))
            .map_err(|error| *error)
    }

    fn publish_admitted_mutations_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_mutation_in_operation(Some(context))
            .map_err(|error| *error)?;
        // Function-local frames own a private current workspace. They must
        // not publish a pending mutation from their caller's projection.
        if self.publish {
            self.canonical_for_operation()
                .map_err(|error| *error)?
                .ok_or_else(native_workspace_closed)?
                .publish_admitted_mutations_with_context(context)?;
        }
        Ok(())
    }

    fn workspace_revision(
        &self,
    ) -> Result<Option<runmat_runtime::workspace::WorkspaceRevision>, runmat_runtime::RuntimeError>
    {
        self.ensure_effect_in_operation(None)
            .map_err(|error| *error)?;
        self.canonical_for_operation()
            .map_err(|error| *error)?
            .ok_or_else(native_workspace_closed)?
            .workspace_revision()
    }

    fn workspace_revision_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Option<runmat_runtime::workspace::WorkspaceRevision>, runmat_runtime::RuntimeError>
    {
        self.ensure_effect_in_operation(Some(context))
            .map_err(|error| *error)?;
        self.canonical_for_operation()
            .map_err(|error| *error)?
            .ok_or_else(native_workspace_closed)?
            .workspace_revision_with_context(context)
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), runmat_runtime::RuntimeError> {
        self.assign_in_operation(None, name, value)
            .map_err(|error| *error)
    }

    fn assign_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        name: &str,
        value: Value,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.assign_in_operation(Some(context), name, value)
            .map_err(|error| *error)
    }

    fn clear(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.clear_in_operation(None).map_err(|error| *error)
    }

    fn clear_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.clear_in_operation(Some(context))
            .map_err(|error| *error)
    }

    fn declare_global_binding(
        &self,
        slot: usize,
        name: &str,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.declare_global_in_operation(None, slot, name)
            .map_err(|error| *error)
    }

    fn declare_global_binding_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
        name: &str,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.declare_global_in_operation(Some(context), slot, name)
            .map_err(|error| *error)
    }

    fn declare_legacy_global_binding_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        slot: usize,
        name: &str,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.declare_legacy_global_in_operation(context, slot, name)
            .map_err(|error| *error)
    }

    fn bind_base_global_target_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        name: &str,
        target: &GlobalBindingTarget,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.bind_global_target_in_operation(context, name, target)
            .map_err(|error| *error)
    }

    fn remove(&self, name: &str) -> Result<(), runmat_runtime::RuntimeError> {
        self.remove_in_operation(None, name).map_err(|error| *error)
    }

    fn remove_with_context(
        &self,
        context: &runmat_runtime::context::RuntimeContext,
        name: &str,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.remove_in_operation(Some(context), name)
            .map_err(|error| *error)
    }
}

/// Called only after the workspace effect has committed and its RefCell
/// borrows have been released. A failure reports partial publication.
fn retire_native_lexical_bindings(
    environment: &runmat_value::LexicalEnvironment,
    bindings: impl IntoIterator<Item = runmat_native_codegen::NativeLexicalBinding>,
) -> Result<(), Box<runmat_runtime::RuntimeError>> {
    for binding in bindings {
        match binding.kind {
            runmat_native_codegen::NativeLexicalBindingKind::Owned => {
                runmat_runtime::call::lexical::remove_owned_binding(environment, binding.binding)?;
            }
            runmat_native_codegen::NativeLexicalBindingKind::Inherited => {
                runmat_runtime::call::lexical::remove_binding(environment, binding.binding)?;
            }
            runmat_native_codegen::NativeLexicalBindingKind::Transit => {}
        }
    }
    Ok(())
}

/// A global workspace write may already have committed before lexical
/// publication; a lexical-only write instead has this cell as its sole value
/// authority. Preserve that distinction in the typed failure diagnostic.
fn publish_native_lexical_binding(
    environment: &runmat_value::LexicalEnvironment,
    binding: runmat_native_codegen::NativeLexicalBinding,
    value: Value,
    workspace_committed: bool,
) -> Result<(), Box<runmat_runtime::RuntimeError>> {
    let result = match binding.kind {
        runmat_native_codegen::NativeLexicalBindingKind::Owned => {
            runmat_runtime::call::lexical::write_owned_binding(environment, binding.binding, value)
        }
        runmat_native_codegen::NativeLexicalBindingKind::Inherited => {
            runmat_runtime::call::lexical::write_binding(environment, binding.binding, value)
        }
        runmat_native_codegen::NativeLexicalBindingKind::Transit => return Ok(()),
    };
    result.map_err(|error| {
        let failure_origin = error.failure_origin;
        let mut failure = runmat_runtime::runtime_error::semantic_error_with_source(
            "RunMat:Workspace:LexicalPublication",
            if workspace_committed {
                format!("workspace assignment committed before lexical publication: {error}")
            } else {
                format!("lexical workspace assignment was not published: {error}")
            },
            error,
        );
        failure.gpu_gather_retry = runmat_runtime::runtime_error::GpuGatherRetry::Never;
        failure.failure_origin = failure_origin;
        Box::new(failure)
    })
}

fn native_workspace_closed() -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:FrameClosed",
        "native workspace frame is closed",
    )
}

/// Execute semantic workspace declarations and commit effect markers.
///
/// Call-backed workspace and environment markers preserve ordering, source,
/// safepoint, and effect-epoch information. The call performs the mutation, and
/// its completion boundary reconciles externally changed workspace bindings.
/// Replaying the mutation at a marker would apply it twice. Global and persistent declarations
/// are different: they establish storage aliases and therefore execute here.
pub(super) fn execute(
    state: &mut HostState,
    instruction: &NativeInstruction,
    statement: &MirStmtKind,
) -> NativeExecutorResult<bool> {
    match statement {
        MirStmtKind::WorkspaceEffect { effect, bindings } => {
            match effect {
                WorkspaceEffect::MutatesGlobal => {
                    declare_bindings(state, bindings, |state, slot, name| {
                        state.declare_global(slot, name)
                    })?;
                }
                WorkspaceEffect::MutatesPersistent => {
                    declare_bindings(state, bindings, |state, slot, name| {
                        state.declare_persistent(slot, name)
                    })?;
                }
                WorkspaceEffect::None
                | WorkspaceEffect::ReadsWorkspace
                | WorkspaceEffect::CreatesBinding
                | WorkspaceEffect::ClearsBinding
                | WorkspaceEffect::ClearsFunctionCache
                | WorkspaceEffect::LoadsExternalBindings
                | WorkspaceEffect::DynamicEval => {
                    if !bindings.is_empty() {
                        return Err(NativeExecutorError::Host(
                            "call-backed workspace effect unexpectedly names local bindings".into(),
                        ));
                    }
                }
            }
            publish_bindings(state, instruction, bindings)?;
            state.reconcile_workspace()?;
            Ok(true)
        }
        MirStmtKind::EnvironmentEffect(
            EnvironmentEffect::PathMutation
            | EnvironmentEffect::WorkingDirectoryMutation
            | EnvironmentEffect::FunctionCacheInvalidation
            | EnvironmentEffect::DynamicLookupInvalidation,
        ) => {
            if !instruction.outputs.is_empty() {
                return Err(NativeExecutorError::Host(
                    "environment effect marker unexpectedly produces locals".into(),
                ));
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn declare_bindings(
    state: &mut HostState,
    bindings: &[MirLocalId],
    mut declare: impl FnMut(&mut HostState, usize, String) -> NativeExecutorResult<()>,
) -> NativeExecutorResult<()> {
    for local in bindings {
        let metadata =
            state.function.locals.get(local.0).ok_or_else(|| {
                NativeExecutorError::Host("workspace local is out of bounds".into())
            })?;
        let name = metadata.name.clone().ok_or_else(|| {
            NativeExecutorError::Host("workspace declaration local has no semantic name".into())
        })?;
        declare(state, local.0, name)?;
    }
    Ok(())
}

fn publish_bindings(
    state: &mut HostState,
    instruction: &NativeInstruction,
    bindings: &[MirLocalId],
) -> NativeExecutorResult<()> {
    if bindings.len() != instruction.outputs.len() {
        return Err(NativeExecutorError::Host(
            "workspace binding/output arity does not match Native IR".into(),
        ));
    }
    for (local, output) in bindings.iter().zip(&instruction.outputs) {
        let value = state.locals.get(local.0).copied().ok_or_else(|| {
            NativeExecutorError::Host("workspace output local is out of bounds".into())
        })?;
        state.values.insert(output.value, value);
    }
    Ok(())
}
