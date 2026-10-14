use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::{Rc, Weak},
};

use runmat_hir::{EnvironmentEffect, WorkspaceEffect};
use runmat_mir::{MirLocalId, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_runtime::context::RuntimeWorkspaceService;
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
use runmat_runtime::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection};
use runmat_types::BindingId;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

mod rooted_value;
use rooted_value::RootedWorkspaceValue;

/// One assigned interactive binding supplied to a native script invocation.
///
/// Binding identity selects the exact Native IR local. The retained name is
/// the MATLAB workspace identity used by dynamic workspace builtins.
/// The value lives only in `NativeWorkspaceInput::values`; metadata must not
/// duplicate payloads or compare language values to validate caller identity.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeWorkspaceBinding {
    pub binding: BindingId,
    pub name: String,
}

/// Transactional interactive workspace input for one native invocation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NativeWorkspaceInput {
    /// Complete assigned caller workspace, including names not referenced by
    /// this source but visible to dynamic workspace builtins.
    pub values: BTreeMap<String, Value>,
    /// Function-scoped semantic names for every caller binding retained by
    /// the synthetic entrypoint, including bindings initially unassigned.
    pub local_names: BTreeMap<BindingId, String>,
    /// Exact semantic locals that may read or write caller bindings.
    pub bindings: Vec<NativeWorkspaceBinding>,
    /// Semantic caller bindings that resolve through the session-global store.
    /// Values remain owned by that store; this list carries alias identity only.
    pub global_bindings: Vec<NativeWorkspaceBinding>,
}

impl NativeWorkspaceInput {
    pub fn profile_values(&self) -> Vec<Value> {
        self.values.values().cloned().collect()
    }

    pub(super) fn frame_global_bindings(
        &self,
        function: &runmat_native_codegen::NativeFunction,
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
            aliases.declare_named(local.id.0 as usize, binding.name.clone());
        }
        Ok(aliases)
    }
}

/// Assigned workspace values after a successful native invocation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NativeWorkspaceSnapshot {
    pub values: BTreeMap<String, Value>,
    pub global_bindings: BTreeSet<String>,
}

struct NativeWorkspaceFrame {
    values: RefCell<BTreeMap<String, RootedWorkspaceValue>>,
    lexical: RefCell<Option<NativeLexicalWorkspaceBindings>>,
    externally_mutated: Cell<bool>,
    canonical_revision: Cell<Option<runmat_runtime::workspace::WorkspaceRevision>>,
    active: Cell<bool>,
}

#[derive(Clone)]
struct NativeLexicalWorkspaceBindings {
    environment: runmat_value::LexicalEnvironment,
    by_name: BTreeMap<String, runmat_native_codegen::NativeLexicalBinding>,
}

#[derive(Clone, Copy)]
enum NativeWorkspaceMutationOrigin {
    ExecutorStore,
    CanonicalRefresh,
}

pub(super) struct NativeWorkspaceState {
    frame: Rc<NativeWorkspaceFrame>,
    local_names: BTreeMap<usize, String>,
    publish: bool,
    global_aliases: FrameGlobalBindings,
    canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
    session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
}

impl NativeWorkspaceState {
    pub(super) fn new(
        function: &runmat_native_codegen::NativeFunction,
        input: NativeWorkspaceInput,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
    ) -> NativeExecutorResult<Self> {
        let local_names = function
            .locals
            .iter()
            .filter(|local| local.kind == runmat_native_codegen::NativeLocalKind::Binding)
            .filter_map(|local| {
                local
                    .binding
                    .and_then(|binding| input.local_names.get(&binding).cloned())
                    .or_else(|| local.name.clone())
                    .map(|name| (local.id.0 as usize, name))
            })
            .collect::<BTreeMap<_, _>>();
        let mut seen_bindings = BTreeSet::new();
        let values = input.values;
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
            if !values.contains_key(&supplied.name) {
                return Err(NativeExecutorError::Host(format!(
                    "native workspace binding '{}' is missing from the caller snapshot",
                    supplied.name
                )));
            }
        }
        let values = values
            .into_iter()
            .map(|(name, value)| RootedWorkspaceValue::new(value).map(|value| (name, value)))
            .collect::<Result<_, _>>()
            .map_err(NativeExecutorError::Runtime)?;
        Ok(Self {
            frame: Rc::new(NativeWorkspaceFrame {
                values: RefCell::new(values),
                lexical: RefCell::new(None),
                externally_mutated: Cell::new(false),
                canonical_revision: Cell::new(None),
                active: Cell::new(true),
            }),
            local_names,
            publish: true,
            global_aliases,
            canonical,
            session_variables,
        })
    }

    pub(super) fn function_frame(
        function: &runmat_native_codegen::NativeFunction,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
        session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
    ) -> Self {
        let local_names = function
            .locals
            .iter()
            .filter(|local| local.kind != runmat_native_codegen::NativeLocalKind::Temporary)
            .filter_map(|local| local.name.clone().map(|name| (local.id.0 as usize, name)))
            .collect();
        Self {
            frame: Rc::new(NativeWorkspaceFrame {
                values: RefCell::new(BTreeMap::new()),
                lexical: RefCell::new(None),
                externally_mutated: Cell::new(false),
                canonical_revision: Cell::new(None),
                active: Cell::new(true),
            }),
            local_names,
            publish: false,
            global_aliases,
            canonical,
            session_variables,
        }
    }

    pub(super) fn service(&self) -> Rc<dyn RuntimeWorkspaceService> {
        Rc::new(NativeWorkspaceService {
            frame: Rc::downgrade(&self.frame),
            global_aliases: self.global_aliases.clone(),
            local_names: self.local_names.clone(),
            canonical: self.canonical.as_ref().map(Rc::downgrade),
            publish: self.publish,
            session_variables: self.session_variables.clone(),
        })
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
                .get(&(binding.local.0 as usize))
                .ok_or_else(|| {
                    NativeExecutorError::Host("native lexical binding has no workspace name".into())
                })?;
            if by_name.insert(name.clone(), *binding).is_some() {
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

    pub(super) fn mark_reconciled(&self) {
        self.frame.externally_mutated.set(false);
    }

    pub(super) fn initial_value(&self, slot: usize) -> Option<Value> {
        self.local_names.get(&slot).and_then(|name| {
            self.frame
                .values
                .borrow()
                .get(name)
                .map(|entry| entry.value.clone())
        })
    }

    pub(super) fn initialize_local(&self, slot: usize, value: Value) -> NativeExecutorResult<()> {
        let Some(name) = self.local_names.get(&slot) else {
            return Ok(());
        };
        let rooted = RootedWorkspaceValue::new(value).map_err(NativeExecutorError::Runtime)?;
        let mut values = self
            .frame
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = values.insert(name.clone(), rooted);
        drop(values);
        drop(previous);
        Ok(())
    }

    pub(super) fn synchronize_local(&self, slot: usize, value: Value) -> NativeExecutorResult<()> {
        self.synchronize_local_with_origin(
            slot,
            value,
            NativeWorkspaceMutationOrigin::ExecutorStore,
        )
    }

    pub(super) fn refresh_local(&self, slot: usize, value: Value) -> NativeExecutorResult<()> {
        self.synchronize_local_with_origin(
            slot,
            value,
            NativeWorkspaceMutationOrigin::CanonicalRefresh,
        )
    }

    /// Mirror an observed shared-cell value into the named frame view. This
    /// is a read, not an executor store or another canonical effect.
    pub(super) fn observe_shared_local(
        &self,
        slot: usize,
        value: Option<Value>,
    ) -> NativeExecutorResult<()> {
        let Some(name) = self.local_names.get(&slot) else {
            return Ok(());
        };
        let rooted = value
            .map(RootedWorkspaceValue::new)
            .transpose()
            .map_err(NativeExecutorError::Runtime)?;
        let mut values = self
            .frame
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = if let Some(rooted) = rooted {
            values.insert(name.clone(), rooted)
        } else {
            values.remove(name)
        };
        drop(values);
        drop(previous);
        Ok(())
    }

    fn synchronize_local_with_origin(
        &self,
        slot: usize,
        value: Value,
        origin: NativeWorkspaceMutationOrigin,
    ) -> NativeExecutorResult<()> {
        if let Some(name) = self.local_names.get(&slot) {
            let rooted =
                RootedWorkspaceValue::new(value.clone()).map_err(NativeExecutorError::Runtime)?;
            self.frame
                .values
                .try_borrow_mut()
                .map(drop)
                .map_err(workspace_borrow_error)?;
            if let Some(canonical) = self.canonical.as_ref() {
                if self.publish && !self.global_aliases.is_bound(slot, name) {
                    match origin {
                        NativeWorkspaceMutationOrigin::ExecutorStore => {
                            canonical.assign_executor_slot(slot, name, value.clone())?;
                        }
                        NativeWorkspaceMutationOrigin::CanonicalRefresh => {
                            canonical.assign(name, value.clone())?;
                        }
                    }
                } else {
                    canonical.ensure_mutation_authority()?;
                }
            }
            let mut values = self
                .frame
                .values
                .try_borrow_mut()
                .map_err(workspace_borrow_error)?;
            let previous = values.insert(name.clone(), rooted);
            drop(values);
            drop(previous);
        }
        Ok(())
    }

    pub(super) fn local_values(&self) -> Vec<(usize, Option<Value>)> {
        let values = self.frame.values.borrow();
        self.local_names
            .iter()
            .map(|(slot, name)| (*slot, values.get(name).map(|entry| entry.value.clone())))
            .collect()
    }

    pub(super) fn snapshot(&self) -> Option<NativeWorkspaceSnapshot> {
        self.publish.then(|| NativeWorkspaceSnapshot {
            values: self
                .frame
                .values
                .borrow()
                .iter()
                .map(|(name, entry)| (name.clone(), entry.value.clone()))
                .collect(),
            global_bindings: self
                .global_aliases
                .named_bindings()
                .into_iter()
                .map(|(_, name)| name)
                .collect(),
        })
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

fn workspace_borrow_error(error: std::cell::BorrowMutError) -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "NativeWorkspaceBorrow",
        format!("native workspace mutation is reentrant: {error}"),
    )
}

fn workspace_read_borrow_error(error: std::cell::BorrowError) -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "NativeWorkspaceBorrow",
        format!("native workspace read is reentrant: {error}"),
    )
}

#[cfg(not(test))]
fn missing_session_variable_authority() -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:SessionAuthorityMissing",
        "native workspace service has no session-variable authority",
    )
}

struct NativeWorkspaceService {
    frame: Weak<NativeWorkspaceFrame>,
    global_aliases: FrameGlobalBindings,
    local_names: BTreeMap<usize, String>,
    canonical: Option<Weak<dyn RuntimeWorkspaceService>>,
    publish: bool,
    session_variables: Option<runmat_runtime::workspace::session::SessionVariableAuthority>,
}

struct NativeWorkspaceClosed;

impl From<NativeWorkspaceClosed> for runmat_runtime::RuntimeError {
    fn from(_: NativeWorkspaceClosed) -> Self {
        native_workspace_closed()
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

    fn canonical(&self) -> Option<Rc<dyn RuntimeWorkspaceService>> {
        self.canonical.as_ref().and_then(Weak::upgrade)
    }
}

impl RuntimeWorkspaceService for NativeWorkspaceService {
    fn detach_selected(
        &self,
        selection: &WorkspaceDetachSelection,
    ) -> Result<DetachedWorkspaceValues, runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        let frame = self.frame()?;
        let mut values = frame
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let detached = DetachedWorkspaceValues::new(
            selection.select(
                values
                    .iter()
                    .map(|(name, entry)| (name.clone(), entry.value.clone()))
                    .collect(),
            ),
        )
        .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?;
        // The frame borrow prevents a reentrant writer between the canonical
        // publication and the local retirement. No await or user callback is
        // permitted until both owners have released these exact bindings.
        let _retired_canonical = if self.publish {
            if let Some(canonical) = self.canonical() {
                let delegated = match selection {
                    WorkspaceDetachSelection::All => WorkspaceDetachSelection::All,
                    WorkspaceDetachSelection::Names(names) => {
                        WorkspaceDetachSelection::Names(names.clone())
                    }
                    WorkspaceDetachSelection::Except(_) => WorkspaceDetachSelection::Names(
                        detached
                            .entries()
                            .iter()
                            .map(|(name, _)| name.clone())
                            .collect(),
                    ),
                };
                Some(canonical.detach_selected(&delegated)?)
            } else {
                None
            }
        } else {
            None
        };
        match selection {
            WorkspaceDetachSelection::All => {
                values.clear();
                self.global_aliases.unbind_all();
            }
            WorkspaceDetachSelection::Names(names) => {
                for name in names {
                    values.remove(name);
                    self.global_aliases.unbind_name(name);
                }
            }
            WorkspaceDetachSelection::Except(_) => {
                for (name, _) in detached.entries() {
                    values.remove(name);
                    self.global_aliases.unbind_name(name);
                }
            }
        }
        drop(values);
        frame.externally_mutated.set(true);
        Ok(detached)
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let frame = self.frame.upgrade()?;
        frame.active.get().then_some(())?;
        for entry in frame.values.try_borrow().ok()?.values() {
            visitor(&entry.value)?;
        }
        Some(())
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        let frame = self.frame.upgrade()?;
        frame.active.get().then_some(())?;
        let values = frame.values.try_borrow().ok()?;
        values.get(name).map(|entry| entry.value.clone())
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        let Some(frame) = self.frame.upgrade() else {
            return Vec::new();
        };
        if !frame.active.get() {
            return Vec::new();
        }
        frame
            .values
            .try_borrow()
            .map(|values| {
                values
                    .iter()
                    .map(|(name, entry)| (name.clone(), entry.value.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn global_names(&self) -> Vec<String> {
        let Some(frame) = self.frame.upgrade() else {
            return Vec::new();
        };
        if !frame.active.get() {
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
        self.ensure_effect_authority()?;
        let frame = self.frame()?;
        let values = frame
            .values
            .try_borrow()
            .map_err(workspace_read_borrow_error)?;
        Ok(values.get(name).map(|entry| entry.value.clone()))
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        let frame = self.frame()?;
        let values = frame
            .values
            .try_borrow()
            .map_err(workspace_read_borrow_error)?;
        Ok(values
            .iter()
            .map(|(name, entry)| (name.clone(), entry.value.clone()))
            .collect())
    }

    fn try_global_names(&self) -> Result<Vec<String>, runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        match self.session_variables.as_ref() {
            Some(authority) => authority.global_names(),
            #[cfg(test)]
            None => Ok(runmat_runtime::workspace::session::global_names()),
            #[cfg(not(test))]
            None => Err(missing_session_variable_authority()),
        }
    }

    fn ensure_effect_authority(&self) -> Result<(), runmat_runtime::RuntimeError> {
        let frame = self.frame()?;
        if let Some(canonical) = self.canonical() {
            canonical.ensure_effect_authority()?;
            if !self.publish {
                return Ok(());
            }
            let revision = canonical.workspace_revision()?;
            if revision.is_some() && revision == frame.canonical_revision.get() {
                return Ok(());
            }
            let canonical_values = canonical.try_snapshot()?;
            let global_values = {
                let values = frame
                    .values
                    .try_borrow()
                    .map_err(workspace_read_borrow_error)?;
                self.local_names
                    .iter()
                    .filter(|(slot, name)| self.global_aliases.is_bound(**slot, name))
                    .filter_map(|(_, name)| {
                        values
                            .get(name)
                            .map(|entry| (name.clone(), entry.value.clone()))
                    })
                    .collect::<Vec<_>>()
            };
            let refreshed = canonical_values
                .into_iter()
                .chain(global_values)
                .map(|(name, value)| {
                    RootedWorkspaceValue::new(value)
                        .map(|value| (name, value))
                        .map_err(|error| *error)
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            let mut values = frame
                .values
                .try_borrow_mut()
                .map_err(workspace_borrow_error)?;
            let previous = std::mem::replace(&mut *values, refreshed);
            drop(values);
            frame.externally_mutated.set(true);
            frame.canonical_revision.set(revision);
            drop(previous);
        }
        Ok(())
    }

    fn ensure_mutation_authority(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.frame()?;
        if let Some(canonical) = self.canonical() {
            canonical.ensure_mutation_authority()?;
        }
        Ok(())
    }

    fn workspace_revision(
        &self,
    ) -> Result<Option<runmat_runtime::workspace::WorkspaceRevision>, runmat_runtime::RuntimeError>
    {
        self.ensure_effect_authority()?;
        Ok(self.frame()?.canonical_revision.get())
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        let publish_global = self
            .local_names
            .iter()
            .any(|(slot, local)| local == name && self.global_aliases.is_bound(*slot, name));
        let global_value = publish_global.then(|| value.clone());
        let rooted = RootedWorkspaceValue::new(value).map_err(|error| *error)?;
        let shared_value = rooted.value.clone();
        let frame = self.frame()?;
        let lexical = frame
            .lexical
            .try_borrow()
            .map_err(workspace_read_borrow_error)?
            .as_ref()
            .and_then(|owner| {
                owner
                    .by_name
                    .get(name)
                    .copied()
                    .map(|binding| (owner.environment.clone(), binding))
            });
        frame
            .values
            .try_borrow_mut()
            .map(drop)
            .map_err(workspace_borrow_error)?;
        if self.publish {
            if let Some(canonical) = self.canonical() {
                if publish_global {
                    canonical.remove(name)?;
                } else {
                    canonical.assign(name, rooted.value.clone())?;
                }
            }
        }
        let mut values = frame
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = values.insert(name.to_string(), rooted);
        drop(values);
        if let Some(value) = global_value {
            match self.session_variables.as_ref() {
                Some(authority) => authority.store_global(name, value)?,
                #[cfg(test)]
                None => runmat_runtime::workspace::session::store_global_named(name, value),
                #[cfg(not(test))]
                None => return Err(missing_session_variable_authority()),
            }
        }
        frame.externally_mutated.set(true);
        drop(previous);
        if let Some((environment, binding)) = lexical {
            publish_native_lexical_binding(&environment, binding, shared_value)
                .map_err(|error| *error)?;
        }
        Ok(())
    }

    fn clear(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        let frame = self.frame()?;
        let lexical = frame
            .lexical
            .try_borrow()
            .map_err(workspace_read_borrow_error)?
            .clone();
        frame
            .values
            .try_borrow_mut()
            .map(drop)
            .map_err(workspace_borrow_error)?;
        if self.publish {
            if let Some(canonical) = self.canonical() {
                canonical.clear()?;
            }
        }
        let mut values = frame
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = std::mem::take(&mut *values);
        drop(values);
        frame.externally_mutated.set(true);
        self.global_aliases.unbind_all();
        drop(previous);
        if let Some(lexical) = lexical {
            retire_native_lexical_bindings(&lexical.environment, lexical.by_name.values().copied())
                .map_err(|error| *error)?;
        }
        Ok(())
    }

    fn declare_global_binding(
        &self,
        slot: usize,
        name: &str,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_mutation_authority()?;
        if self.publish {
            if let Some(canonical) = self.canonical() {
                canonical.bind_base_global(name)?;
            }
        }
        self.global_aliases.declare_named(slot, name.to_string());
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        let frame = self.frame()?;
        let lexical = frame
            .lexical
            .try_borrow()
            .map_err(workspace_read_borrow_error)?
            .as_ref()
            .and_then(|owner| {
                owner
                    .by_name
                    .get(name)
                    .copied()
                    .map(|binding| (owner.environment.clone(), binding))
            });
        frame
            .values
            .try_borrow_mut()
            .map(drop)
            .map_err(workspace_borrow_error)?;
        if self.publish {
            if let Some(canonical) = self.canonical() {
                canonical.unbind_base_global(name)?;
            }
        }
        let mut values = frame
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = values.remove(name);
        drop(values);
        frame.externally_mutated.set(true);
        self.global_aliases.unbind_name(name);
        drop(previous);
        if let Some((environment, binding)) = lexical {
            retire_native_lexical_bindings(&environment, std::iter::once(binding))
                .map_err(|error| *error)?;
        }
        Ok(())
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

/// The canonical and local workspace write has committed before this call.
/// Report an explicit partial publication error if the shared cell rejects it.
fn publish_native_lexical_binding(
    environment: &runmat_value::LexicalEnvironment,
    binding: runmat_native_codegen::NativeLexicalBinding,
    value: Value,
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
            format!("workspace assignment committed before lexical publication: {error}"),
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
