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
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => rvalue_contract(value, summaries),
        MirStmtKind::CaptureSequence { source, .. } => {
            expansion_source_contract(source, summaries, &[])
        }
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
                &BTreeMap::new(),
                &literals,
                summaries,
                OutputSelection::new(call.requested_outputs),
            );
            effects = inference.effects;
            extend_declared_call_effects(&mut effects, &call.effects);
            capabilities = inference.capabilities;
        }
        MirRvalue::Index { base, .. } => {
            if let runmat_types::ValueKindFact::Callable(callable) =
                super::value::operand_fact_from_values(base, facts, summaries).kind
            {
                capabilities.0.extend(callable.capabilities.0);
            }
        }
        MirRvalue::SubscriptChain(chain) => {
            let inferred = crate::analysis::dataflow::infer_subscript_chain_contract(chain, facts);
            effects.0.extend(inferred.effects.0);
            capabilities.0.extend(inferred.capabilities.0);
            if let crate::analysis::dataflow::SubscriptPathDisposition::UnresolvedDynamic {
                step: _step,
            } = inferred.disposition
            {
                // Only an unresolved object-protocol boundary is effectful. Fully
                // proven numeric/cell/struct paths retain ordinary default-index
                // purity and placement eligibility.
                effects.0.insert(EffectKind::Unknown);
                effects.0.insert(EffectKind::MaySuspend);
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
    value.visit_direct_expression_steps_dyn(&mut |steps| {
        let (nested_effects, nested_capabilities) =
            expression_steps_contract(steps, summaries, facts);
        effects.0.extend(nested_effects.0);
        capabilities.0.extend(nested_capabilities.0);
    });
    (effects, capabilities)
}

fn expansion_source_contract(
    source: &crate::MirExpansionSource,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
) -> (EffectSet, CapabilitySet) {
    let mut effects = EffectSet([EffectKind::Unknown].into_iter().collect());
    let mut capabilities = CapabilitySet::default();
    source.visit_direct_expression_steps_dyn(&mut |steps| {
        let (nested_effects, nested_capabilities) =
            expression_steps_contract(steps, summaries, facts);
        effects.0.extend(nested_effects.0);
        capabilities.0.extend(nested_capabilities.0);
    });
    (effects, capabilities)
}

fn expression_steps_contract(
    steps: &[crate::MirExpressionStep],
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
) -> (EffectSet, CapabilitySet) {
    let mut nested_facts = facts.to_vec();
    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    for step in steps {
        match step {
            crate::MirExpressionStep::Let { local, value, .. } => {
                let (step_effects, step_capabilities) =
                    rvalue_contract_with_facts(value, summaries, &nested_facts);
                effects.0.extend(step_effects.0);
                capabilities.0.extend(step_capabilities.0);
                let fact = crate::analysis::dataflow::simple_rvalue_fact(value, &nested_facts);
                if let Some(slot) = nested_facts.get_mut(local.0) {
                    *slot = Some(fact);
                }
            }
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                let (step_effects, step_capabilities) =
                    expansion_source_contract(source, summaries, &nested_facts);
                effects.0.extend(step_effects.0);
                capabilities.0.extend(step_capabilities.0);
            }
        }
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
        MirStmtKind::CaptureSequence { source, .. } => {
            expansion_source_contract(source, summaries, facts)
        }
        _ => statement_contract(statement, summaries),
    }
}

#[cfg(test)]
mod dotted_argument_tests {
    use super::*;
    use crate::{
        MirCallArg, MirCallArgumentRegion, MirDottedInvokeArgument, MirExpressionStep, MirLocalId,
        MirOperand, MirSubscriptChain, MirSubscriptStep,
    };

    #[test]
    fn preparation_future_capability_is_visible_inside_dotted_invocation() {
        let chain = MirSubscriptChain {
            root: MirOperand::Local(MirLocalId(0)),
            steps: vec![MirSubscriptStep::DottedInvoke {
                semantic_fallback: None,
                member: runmat_hir::MemberName("invoke".to_owned()),
                arguments: vec![MirDottedInvokeArgument::Call(MirCallArgumentRegion {
                    steps: vec![MirExpressionStep::Let {
                        local: MirLocalId(1),
                        value: MirRvalue::Future {
                            function: FunctionId(7),
                            args: Vec::new(),
                            syntax: runmat_hir::CallSyntax::Plain,
                            requested_outputs: runmat_hir::RequestedOutputCount::One,
                        },
                        span: runmat_hir::Span::default(),
                    }],
                    argument: MirCallArg::Single(MirOperand::Local(MirLocalId(1))),
                    span: runmat_hir::Span::default(),
                })],
            }],
            sequence_use: runmat_types::SequenceUse::RequireSingle,
            context: runmat_types::ObjectIndexingContext::Expression,
        };
        let (effects, capabilities) =
            rvalue_contract(&MirRvalue::SubscriptChain(chain), &BTreeMap::new());
        assert!(effects.0.contains(&EffectKind::MaySuspend));
        assert!(capabilities
            .0
            .contains(&CapabilityRequirement::ParallelRuntime));
    }
}
