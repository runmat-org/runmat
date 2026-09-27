use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use runmat_hir::{EnvironmentEffect, WorkspaceEffect};
use runmat_mir::{MirLocalId, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_runtime::context::RuntimeWorkspaceService;
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

#[derive(Clone)]
pub(super) struct NativeWorkspaceState {
    values: Rc<RefCell<BTreeMap<String, RootedWorkspaceValue>>>,
    externally_mutated: Rc<Cell<bool>>,
    local_names: BTreeMap<usize, String>,
    publish: bool,
}

impl NativeWorkspaceState {
    pub(super) fn new(
        function: &runmat_native_codegen::NativeFunction,
        input: NativeWorkspaceInput,
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
        })
    }

    pub(super) fn function_frame(function: &runmat_native_codegen::NativeFunction) -> Self {
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
        }
    }

    pub(super) fn service(&self) -> Rc<dyn RuntimeWorkspaceService> {
        Rc::new(NativeWorkspaceService {
            values: Rc::clone(&self.values),
            externally_mutated: Rc::clone(&self.externally_mutated),
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
            let rooted = RootedWorkspaceValue::new(value).map_err(NativeExecutorError::Runtime)?;
            let previous = self
                .values
                .try_borrow_mut()
                .map_err(workspace_borrow_error)?
                .insert(name.clone(), rooted);
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

struct NativeWorkspaceService {
    values: Rc<RefCell<BTreeMap<String, RootedWorkspaceValue>>>,
    externally_mutated: Rc<Cell<bool>>,
}

impl RuntimeWorkspaceService for NativeWorkspaceService {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for entry in self.values.try_borrow().ok()?.values() {
            visitor(&entry.value)?;
        }
        Some(())
    }
    fn lookup(&self, name: &str) -> Option<Value> {
        self.values
            .borrow()
            .get(name)
            .map(|entry| entry.value.clone())
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        self.values
            .borrow()
            .iter()
            .map(|(name, entry)| (name.clone(), entry.value.clone()))
            .collect()
    }

    fn global_names(&self) -> Vec<String> {
        runmat_runtime::workspace::session::global_names()
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), runmat_runtime::RuntimeError> {
        let rooted = RootedWorkspaceValue::new(value).map_err(|error| *error)?;
        let previous = self
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?
            .insert(name.to_string(), rooted);
        self.externally_mutated.set(true);
        drop(previous);
        Ok(())
    }

    fn clear(&self) -> Result<(), runmat_runtime::RuntimeError> {
        let previous = std::mem::take(
            &mut *self
                .values
                .try_borrow_mut()
                .map_err(workspace_borrow_error)?,
        );
        self.externally_mutated.set(true);
        drop(previous);
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<(), runmat_runtime::RuntimeError> {
        let previous = self
            .values
            .try_borrow_mut()
            .map_err(workspace_borrow_error)?
            .remove(name);
        self.externally_mutated.set(true);
        drop(previous);
        Ok(())
    }
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
