use std::collections::BTreeMap;

use runmat_hir::FunctionId;
use runmat_types::{
    CapabilityRequirement, CapabilitySet, EffectKind, EffectSet, LiteralValue, OutputSelection,
    ValueFact,
};

use crate::{MirRvalue, MirStmt, MirStmtKind};

use crate::analysis::engine::FlowState;

use super::{infer_mir_call, FunctionSummary};

pub(crate) fn apply_rvalue_contract(
    value: &MirRvalue,
    state: &mut FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) {
    let facts = state.value_facts();
    let (effects, capabilities) = rvalue_contract_with_facts(value, summaries, &facts);
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
        MirStmtKind::PlaceMutation(mutation)
            if mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::Overloaded
                || mutation.shape_policy == runmat_hir::AssignmentShapePolicy::Overloaded
                || mutation.kind == runmat_hir::PlaceMutationKind::MemberAssign =>
        {
            (
                EffectSet([EffectKind::Unknown].into_iter().collect()),
                CapabilitySet::default(),
            )
        }
        MirStmtKind::PlaceMutation(_) => (EffectSet::default(), CapabilitySet::default()),
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
    rvalue_contract_with_facts(value, summaries, &[])
}

pub(crate) fn rvalue_contract_with_facts(
    value: &MirRvalue,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
) -> (EffectSet, CapabilitySet) {
    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    match value {
        MirRvalue::Call(call) => {
            let literals = vec![LiteralValue::Unknown; call.args.len()];
            let inference = infer_mir_call(
                call,
                facts,
                &literals,
                summaries,
                OutputSelection::new(call.requested_outputs),
            );
            let mut effects = inference.effects;
            extend_declared_call_effects(&mut effects, &call.effects);
            return (effects, inference.capabilities);
        }
        MirRvalue::Index { base, .. } => {
            if let runmat_types::ValueKindFact::Callable(callable) =
                super::value::operand_fact_from_values(base, facts, summaries).kind
            {
                capabilities.0.extend(callable.capabilities.0);
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
                    statement_contract_with_facts(statement, summaries, facts);
                effects.0.extend(nested_effects.0);
                capabilities.0.extend(nested_capabilities.0);
            }
        }
        _ => {}
    }
    (effects, capabilities)
}

fn extend_declared_call_effects(
    effects: &mut EffectSet,
    declared: &runmat_builtins::BuiltinEffects,
) {
    for (present, effect) in [
        (declared.workspace, EffectKind::WorkspaceWrite),
        (declared.environment, EffectKind::EnvironmentWrite),
        (declared.filesystem, EffectKind::FilesystemRead),
        (declared.network, EffectKind::Network),
        (declared.ui, EffectKind::UserInterface),
        (declared.random, EffectKind::Randomness),
        (declared.time, EffectKind::Clock),
        (declared.host_callback, EffectKind::HostCallback),
        (declared.unknown, EffectKind::Unknown),
    ] {
        if present {
            effects.0.insert(effect);
        }
    }
}

pub(crate) fn statement_contract_with_facts(
    statement: &MirStmt,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
) -> (EffectSet, CapabilitySet) {
    match &statement.kind {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::Expr(value) => rvalue_contract_with_facts(value, summaries, facts),
        _ => statement_contract(statement, summaries),
    }
}
