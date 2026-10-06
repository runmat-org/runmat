use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use runmat_hir::{EnvironmentEffect, WorkspaceEffect};
use runmat_mir::{MirLocalId, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_runtime::context::RuntimeWorkspaceService;
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
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
}

impl NativeWorkspaceInput {
    pub fn profile_values(&self) -> Vec<Value> {
        self.values.values().cloned().collect()
    }
}

/// Assigned workspace values after a successful native invocation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NativeWorkspaceSnapshot {
    pub values: BTreeMap<String, Value>,
}

pub(super) struct NativeWorkspaceState {
    values: Rc<RefCell<BTreeMap<String, RootedWorkspaceValue>>>,
    externally_mutated: Rc<Cell<bool>>,
    local_names: BTreeMap<usize, String>,
    publish: bool,
    global_aliases: FrameGlobalBindings,
    canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
    canonical_revision: Rc<Cell<Option<runmat_runtime::workspace::WorkspaceRevision>>>,
    active: Rc<Cell<bool>>,
}

impl NativeWorkspaceState {
    pub(super) fn new(
        function: &runmat_native_codegen::NativeFunction,
        input: NativeWorkspaceInput,
        global_aliases: FrameGlobalBindings,
        canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
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
            values: Rc::new(RefCell::new(values)),
            externally_mutated: Rc::new(Cell::new(false)),
            local_names,
            publish: true,
            global_aliases,
            canonical,
            canonical_revision: Rc::new(Cell::new(None)),
            active: Rc::new(Cell::new(true)),
        })
    }

    pub(super) fn function_frame(
        function: &runmat_native_codegen::NativeFunction,
        global_aliases: FrameGlobalBindings,
    ) -> Self {
        let local_names = function
            .locals
            .iter()
            .filter(|local| local.kind != runmat_native_codegen::NativeLocalKind::Temporary)
            .filter_map(|local| local.name.clone().map(|name| (local.id.0 as usize, name)))
            .collect();
        Self {
            values: Rc::new(RefCell::new(BTreeMap::new())),
            externally_mutated: Rc::new(Cell::new(false)),
            local_names,
            publish: false,
            global_aliases,
            canonical: None,
            canonical_revision: Rc::new(Cell::new(None)),
            active: Rc::new(Cell::new(true)),
        }
    }

    pub(super) fn service(&self) -> Rc<dyn RuntimeWorkspaceService> {
        Rc::new(NativeWorkspaceService {
            values: Rc::clone(&self.values),
            externally_mutated: Rc::clone(&self.externally_mutated),
            global_aliases: self.global_aliases.clone(),
            local_names: self.local_names.clone(),
            canonical: self.canonical.clone(),
            canonical_revision: Rc::clone(&self.canonical_revision),
            active: Rc::clone(&self.active),
        })
    }

    pub(super) fn needs_reconciliation(&self) -> bool {
        self.externally_mutated.get()
    }

    pub(super) fn mark_reconciled(&self) {
        self.externally_mutated.set(false);
    }

    pub(super) fn initial_value(&self, slot: usize) -> Option<Value> {
        self.local_names.get(&slot).and_then(|name| {
            self.values
                .borrow()
                .get(name)
                .map(|entry| entry.value.clone())
        })
    }

    pub(super) fn synchronize_local(&self, slot: usize, value: Value) -> NativeExecutorResult<()> {
        if let Some(name) = self.local_names.get(&slot) {
            let rooted =
                RootedWorkspaceValue::new(value.clone()).map_err(NativeExecutorError::Runtime)?;
            self.values
                .try_borrow_mut()
                .map(drop)
                .map_err(workspace_borrow_error)?;
            if self.publish && !self.global_aliases.is_bound(slot, name) {
                if let Some(canonical) = self.canonical.as_ref() {
                    canonical
                        .assign(name, value.clone())
                        .map_err(NativeExecutorError::Runtime)?;
                }
            }
            let mut values = self
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
        let values = self.values.borrow();
        self.local_names
            .iter()
            .map(|(slot, name)| (*slot, values.get(name).map(|entry| entry.value.clone())))
            .collect()
    }

    pub(super) fn snapshot(&self) -> Option<NativeWorkspaceSnapshot> {
        self.publish.then(|| NativeWorkspaceSnapshot {
            values: self
                .values
                .borrow()
                .iter()
                .map(|(name, entry)| (name.clone(), entry.value.clone()))
                .collect(),
        })
    }
}

impl Drop for NativeWorkspaceState {
    fn drop(&mut self) {
        self.active.set(false);
    }
}

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod tests;

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

struct NativeWorkspaceService {
    values: Rc<RefCell<BTreeMap<String, RootedWorkspaceValue>>>,
    externally_mutated: Rc<Cell<bool>>,
    global_aliases: FrameGlobalBindings,
    local_names: BTreeMap<usize, String>,
    canonical: Option<Rc<dyn RuntimeWorkspaceService>>,
    canonical_revision: Rc<Cell<Option<runmat_runtime::workspace::WorkspaceRevision>>>,
    active: Rc<Cell<bool>>,
}

impl NativeWorkspaceService {
    fn ensure_active(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.active
            .get()
            .then_some(())
            .ok_or_else(native_workspace_closed)
    }
}

impl RuntimeWorkspaceService for NativeWorkspaceService {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        self.active.get().then_some(())?;
        for entry in self.values.try_borrow().ok()?.values() {
            visitor(&entry.value)?;
        }
        Some(())
    }
    fn lookup(&self, name: &str) -> Option<Value> {
        self.active.get().then_some(())?;
        self.values
            .borrow()
            .get(name)
            .map(|entry| entry.value.clone())
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        if !self.active.get() {
            return Vec::new();
        }
        self.values
            .borrow()
            .iter()
            .map(|(name, entry)| (name.clone(), entry.value.clone()))
            .collect()
    }

    fn global_names(&self) -> Vec<String> {
        if !self.active.get() {
            return Vec::new();
        }
        runmat_runtime::workspace::session::global_names()
    }

    fn try_lookup(&self, name: &str) -> Result<Option<Value>, runmat_runtime::RuntimeError> {
        self.ensure_active()?;
        Ok(self.lookup(name))
    }

    fn try_snapshot(&self) -> Result<Vec<(String, Value)>, runmat_runtime::RuntimeError> {
        self.ensure_active()?;
        Ok(self.snapshot())
    }

    fn try_global_names(&self) -> Result<Vec<String>, runmat_runtime::RuntimeError> {
        self.ensure_active()?;
        Ok(runmat_runtime::workspace::session::global_names())
    }

    fn ensure_effect_authority(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_active()?;
        if let Some(canonical) = self.canonical.as_ref() {
            canonical.ensure_effect_authority()?;
            let revision = canonical.workspace_revision()?;
            if revision.is_some() && revision == self.canonical_revision.get() {
                return Ok(());
            }
            let canonical_values = canonical.try_snapshot()?;
            let global_values = {
                let values = self
                    .values
                    .try_borrow()
                    .map_err(|error| workspace_read_borrow_error(error))?;
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
            let mut values = self
                .values
                .try_borrow_mut()
                .map_err(workspace_borrow_error)?;
            let previous = std::mem::replace(&mut *values, refreshed);
            drop(values);
            self.externally_mutated.set(true);
            self.canonical_revision.set(revision);
            drop(previous);
        }
        Ok(())
    }

    fn workspace_revision(
        &self,
    ) -> Result<Option<runmat_runtime::workspace::WorkspaceRevision>, runmat_runtime::RuntimeError>
    {
        self.ensure_effect_authority()?;
        Ok(self.canonical_revision.get())
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        let publish_global = self
            .local_names
            .iter()
            .any(|(slot, local)| local == name && self.global_aliases.is_bound(*slot, name));
        let global_value = publish_global.then(|| value.clone());
        let rooted = RootedWorkspaceValue::new(value).map_err(|error| *error)?;
        self.values
            .try_borrow_mut()
            .map(drop)
            .map_err(workspace_borrow_error)?;
        if !publish_global {
            if let Some(canonical) = self.canonical.as_ref() {
                canonical.assign(name, rooted.value.clone())?;
            }
        }
        let mut values = self
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = values.insert(name.to_string(), rooted);
        drop(values);
        if let Some(value) = global_value {
            runmat_runtime::workspace::session::store_global_named(name, value);
        }
        self.externally_mutated.set(true);
        drop(previous);
        Ok(())
    }

    fn clear(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        self.values
            .try_borrow_mut()
            .map(drop)
            .map_err(workspace_borrow_error)?;
        if let Some(canonical) = self.canonical.as_ref() {
            canonical.clear()?;
        }
        let mut values = self
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = std::mem::take(&mut *values);
        drop(values);
        self.externally_mutated.set(true);
        self.global_aliases.unbind_all();
        drop(previous);
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<(), runmat_runtime::RuntimeError> {
        self.ensure_effect_authority()?;
        self.values
            .try_borrow_mut()
            .map(drop)
            .map_err(workspace_borrow_error)?;
        if let Some(canonical) = self.canonical.as_ref() {
            canonical.remove(name)?;
        }
        let mut values = self
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?;
        let previous = values.remove(name);
        drop(values);
        self.externally_mutated.set(true);
        self.global_aliases.unbind_name(name);
        drop(previous);
        Ok(())
    }
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
