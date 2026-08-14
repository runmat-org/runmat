use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use runmat_hir::{EnvironmentEffect, WorkspaceEffect};
use runmat_mir::{MirLocalId, MirStmtKind};
use runmat_native_codegen::NativeInstruction;
use runmat_runtime::context::RuntimeWorkspaceService;
use runmat_types::BindingId;
use runmat_value::Value;

use crate::{JitError, JitResult};

use super::state::HostState;

/// One assigned interactive binding supplied to a native script invocation.
///
/// Binding identity selects the exact Native IR local. The retained name is
/// the MATLAB workspace identity used by dynamic workspace builtins.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeWorkspaceBinding {
    pub binding: BindingId,
    pub name: String,
    pub value: Value,
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
    values: Rc<RefCell<BTreeMap<String, Value>>>,
    local_names: BTreeMap<usize, String>,
    publish: bool,
}

impl NativeWorkspaceState {
    pub(super) fn new(
        function: &runmat_native_codegen::NativeFunction,
        input: NativeWorkspaceInput,
    ) -> JitResult<Self> {
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
        let mut values = input.values;
        for supplied in input.bindings {
            if !seen_bindings.insert(supplied.binding) {
                return Err(JitError::Host(
                    "native workspace input contains a duplicate binding".into(),
                ));
            }
            let local = function
                .locals
                .iter()
                .find(|local| local.binding == Some(supplied.binding))
                .ok_or_else(|| {
                    JitError::Host(format!(
                        "native workspace binding '{}' is unavailable",
                        supplied.name
                    ))
                })?;
            if local.kind != runmat_native_codegen::NativeLocalKind::Binding
                || input.local_names.get(&supplied.binding) != Some(&supplied.name)
            {
                return Err(JitError::Host(format!(
                    "native workspace binding '{}' does not match Native IR",
                    supplied.name
                )));
            }
            if let Some(ambient) = values.get(&supplied.name) {
                if ambient != &supplied.value {
                    return Err(JitError::Host(format!(
                        "native workspace binding '{}' disagrees with the caller snapshot",
                        supplied.name
                    )));
                }
            } else {
                values.insert(supplied.name, supplied.value);
            }
        }
        Ok(Self {
            values: Rc::new(RefCell::new(values)),
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
            local_names,
            publish: false,
        }
    }

    pub(super) fn service(&self) -> Rc<dyn RuntimeWorkspaceService> {
        Rc::new(NativeWorkspaceService {
            values: Rc::clone(&self.values),
        })
    }

    pub(super) fn initial_value(&self, slot: usize) -> Option<Value> {
        self.local_names
            .get(&slot)
            .and_then(|name| self.values.borrow().get(name).cloned())
    }

    pub(super) fn synchronize_local(&self, slot: usize, value: Value) {
        if let Some(name) = self.local_names.get(&slot) {
            self.values.borrow_mut().insert(name.clone(), value);
        }
    }

    pub(super) fn local_values(&self) -> Vec<(usize, Option<Value>)> {
        let values = self.values.borrow();
        self.local_names
            .iter()
            .map(|(slot, name)| (*slot, values.get(name).cloned()))
            .collect()
    }

    pub(super) fn snapshot(&self) -> Option<NativeWorkspaceSnapshot> {
        self.publish.then(|| NativeWorkspaceSnapshot {
            values: self.values.borrow().clone(),
        })
    }
}

struct NativeWorkspaceService {
    values: Rc<RefCell<BTreeMap<String, Value>>>,
}

impl RuntimeWorkspaceService for NativeWorkspaceService {
    fn lookup(&self, name: &str) -> Option<Value> {
        self.values.borrow().get(name).cloned()
    }

    fn snapshot(&self) -> Vec<(String, Value)> {
        self.values
            .borrow()
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    fn global_names(&self) -> Vec<String> {
        runmat_runtime::workspace::session::global_names()
    }

    fn assign(&self, name: &str, value: Value) -> Result<(), runmat_runtime::RuntimeError> {
        self.values.borrow_mut().insert(name.to_string(), value);
        Ok(())
    }

    fn clear(&self) -> Result<(), runmat_runtime::RuntimeError> {
        self.values.borrow_mut().clear();
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<(), runmat_runtime::RuntimeError> {
        self.values.borrow_mut().remove(name);
        Ok(())
    }
}

/// Execute semantic workspace declarations and commit effect markers.
///
/// Call-backed workspace and environment effects describe what the preceding
/// runtime call already performed. Replaying those mutations here would apply
/// them twice; their Native IR instruction instead preserves ordering, source,
/// safepoint, and effect-epoch information. Global and persistent declarations
/// are different: they establish storage aliases and therefore execute here.
pub(super) fn execute(
    state: &mut HostState,
    instruction: &NativeInstruction,
    statement: &MirStmtKind,
) -> JitResult<bool> {
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
                        return Err(JitError::Host(
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
                return Err(JitError::Host(
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
    mut declare: impl FnMut(&mut HostState, usize, String) -> JitResult<()>,
) -> JitResult<()> {
    for local in bindings {
        let metadata = state
            .function
            .locals
            .get(local.0)
            .ok_or_else(|| JitError::Host("workspace local is out of bounds".into()))?;
        let name = metadata.name.clone().ok_or_else(|| {
            JitError::Host("workspace declaration local has no semantic name".into())
        })?;
        declare(state, local.0, name)?;
    }
    Ok(())
}

fn publish_bindings(
    state: &mut HostState,
    instruction: &NativeInstruction,
    bindings: &[MirLocalId],
) -> JitResult<()> {
    if bindings.len() != instruction.outputs.len() {
        return Err(JitError::Host(
            "workspace binding/output arity does not match Native IR".into(),
        ));
    }
    for (local, output) in bindings.iter().zip(&instruction.outputs) {
        let value = state
            .locals
            .get(local.0)
            .copied()
            .ok_or_else(|| JitError::Host("workspace output local is out of bounds".into()))?;
        state.values.insert(output.value, value);
    }
    Ok(())
}
