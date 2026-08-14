use std::collections::BTreeMap;

use runmat_hir::FunctionId;
use runmat_types::{CapabilityRequirement, CapabilitySet, EffectKind, EffectSet};

use crate::{MirRvalue, MirStmt, MirStmtKind};

use crate::analysis::engine::FlowState;

use super::FunctionSummary;

pub(crate) fn apply_rvalue_contract(
    value: &MirRvalue,
    state: &mut FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) {
    let (effects, capabilities) = rvalue_contract(value, summaries);
    state.effects.0.extend(effects.0);
    state.capabilities.0.extend(capabilities.0);
}

pub(crate) fn statement_contract(
    statement: &MirStmt,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> (EffectSet, CapabilitySet) {
    match &statement.kind {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::Expr(value) => rvalue_contract(value, summaries),
        MirStmtKind::PlaceMutation(_) => (
            EffectSet([EffectKind::Unknown].into_iter().collect()),
            CapabilitySet::default(),
        ),
        MirStmtKind::WorkspaceEffect { .. } => (
            EffectSet([EffectKind::WorkspaceWrite].into_iter().collect()),
            CapabilitySet::default(),
        ),
        MirStmtKind::EnvironmentEffect(_) => (
            EffectSet([EffectKind::EnvironmentWrite].into_iter().collect()),
            CapabilitySet::default(),
        ),
    }
}

pub(crate) fn rvalue_contract(
    value: &MirRvalue,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> (EffectSet, CapabilitySet) {
    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    match value {
        MirRvalue::Call(call) => {
            if let Some(function) = bound_function(call) {
                if let Some(summary) = summaries.get(&function) {
                    return (summary.effects.clone(), summary.capabilities.clone());
                }
            }
            if let Some(name) = call_name(call) {
                if let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(&name) {
                    return (entry.contract.effect_set(), entry.contract.capability_set());
                }
            }
            if call.effects.workspace {
                effects.0.insert(EffectKind::WorkspaceWrite);
            }
            if call.effects.environment {
                effects.0.insert(EffectKind::EnvironmentWrite);
            }
            if call.effects.filesystem {
                effects.0.insert(EffectKind::FilesystemRead);
            }
            if call.effects.network {
                effects.0.insert(EffectKind::Network);
            }
            if call.effects.ui {
                effects.0.insert(EffectKind::UserInterface);
            }
            if call.effects.random {
                effects.0.insert(EffectKind::Randomness);
            }
            if call.effects.time {
                effects.0.insert(EffectKind::Clock);
            }
            if call.effects.host_callback {
                effects.0.insert(EffectKind::HostCallback);
            }
            if call.effects.unknown {
                effects.0.insert(EffectKind::Unknown);
            }
        }
        MirRvalue::Future { .. } | MirRvalue::Spawn(_) => {
            effects.0.insert(EffectKind::MaySuspend);
            capabilities
                .0
                .insert(CapabilityRequirement::ParallelRuntime);
        }
        MirRvalue::Distributed(_) | MirRvalue::Collective(_) => {
            capabilities
                .0
                .insert(CapabilityRequirement::DistributedRuntime);
        }
        MirRvalue::ShortCircuit { right_temps, .. } => {
            for statement in right_temps {
                let (nested_effects, nested_capabilities) =
                    statement_contract(statement, summaries);
                effects.0.extend(nested_effects.0);
                capabilities.0.extend(nested_capabilities.0);
            }
        }
        _ => {}
    }
    (effects, capabilities)
}

fn bound_function(call: &crate::MirCall) -> Option<FunctionId> {
    match &call.callee {
        crate::MirCallee::Static(
            runmat_hir::CallableIdentity::BoundFunction(function)
            | runmat_hir::CallableIdentity::AnonymousFunction(function)
            | runmat_hir::CallableIdentity::ExternalFunction { function, .. },
        ) => Some(*function),
        _ => None,
    }
}

fn call_name(call: &crate::MirCall) -> Option<String> {
    match &call.callee {
        crate::MirCallee::Static(identity) => identity.display_name(),
        crate::MirCallee::SuperConstructor { super_class, .. } => Some(super_class.clone()),
        crate::MirCallee::SuperMethod { method, .. } => Some(method.clone()),
        crate::MirCallee::Dynamic(_) => None,
    }
}
