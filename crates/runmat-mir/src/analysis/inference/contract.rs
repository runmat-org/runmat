use std::collections::BTreeMap;

use runmat_hir::FunctionId;
use runmat_types::{CapabilityRequirement, EffectKind};

use crate::MirRvalue;

use crate::analysis::engine::FlowState;

use super::FunctionSummary;

pub(crate) fn apply_rvalue_contract(
    value: &MirRvalue,
    state: &mut FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) {
    match value {
        MirRvalue::Call(call) => {
            if let Some(function) = bound_function(call) {
                if let Some(summary) = summaries.get(&function) {
                    state.effects.0.extend(summary.effects.0.iter().copied());
                    state
                        .capabilities
                        .0
                        .extend(summary.capabilities.0.iter().copied());
                    return;
                }
            }
            if let Some(name) = call_name(call) {
                if let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(&name) {
                    state.effects.0.extend(entry.contract.effect_set().0);
                    state
                        .capabilities
                        .0
                        .extend(entry.contract.capability_set().0);
                    return;
                }
            }
            if call.effects.workspace {
                state.effects.0.insert(EffectKind::WorkspaceWrite);
            }
            if call.effects.environment {
                state.effects.0.insert(EffectKind::EnvironmentWrite);
            }
            if call.effects.filesystem {
                state.effects.0.insert(EffectKind::FilesystemRead);
            }
            if call.effects.network {
                state.effects.0.insert(EffectKind::Network);
            }
            if call.effects.ui {
                state.effects.0.insert(EffectKind::UserInterface);
            }
            if call.effects.random {
                state.effects.0.insert(EffectKind::Randomness);
            }
            if call.effects.time {
                state.effects.0.insert(EffectKind::Clock);
            }
            if call.effects.host_callback {
                state.effects.0.insert(EffectKind::HostCallback);
            }
            if call.effects.unknown {
                state.effects.0.insert(EffectKind::Unknown);
            }
        }
        MirRvalue::Future { .. } | MirRvalue::Spawn(_) => {
            state.effects.0.insert(EffectKind::MaySuspend);
        }
        MirRvalue::Distributed(_) | MirRvalue::Collective(_) => {
            state
                .capabilities
                .0
                .insert(CapabilityRequirement::DistributedRuntime);
        }
        _ => {}
    }
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
