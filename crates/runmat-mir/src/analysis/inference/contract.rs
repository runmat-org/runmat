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
    let (effects, capabilities) = rvalue_operation_contract(value, state, summaries);
    state.effects.0.extend(effects.0);
    state.capabilities.0.extend(capabilities.0);
}

pub(crate) fn rvalue_operation_contract(
    value: &MirRvalue,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> (EffectSet, CapabilitySet) {
    let facts = state.value_facts();
    // Expansion/captured sequences can change argument positions. Keep their
    // call-site literals conservative until the exact expansion is proven.
    let literals = match value {
        MirRvalue::Call(call) => proven_call_literals(call, state),
        _ => Vec::new(),
    };
    rvalue_contract_with_facts_and_literals(
        value,
        summaries,
        &facts,
        state.sequence_facts(),
        Some(&literals),
    )
}

fn proven_call_literals(call: &crate::MirCall, state: &FlowState) -> Vec<LiteralValue> {
    if !call
        .args
        .iter()
        .all(|argument| matches!(argument, crate::MirCallArg::Single(_)))
    {
        return Vec::new();
    }
    call.args
        .iter()
        .filter_map(|argument| argument.operand())
        .map(|operand| super::value::operand_literal(operand, state))
        .collect()
}

pub(crate) fn statement_contract(
    statement: &MirStmt,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> (EffectSet, CapabilitySet) {
    match &statement.kind {
        MirStmtKind::PreparedAssign { value, .. }
        | MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => rvalue_contract(value, summaries),
        MirStmtKind::CaptureSequence { source, .. } => {
            expansion_source_contract(source, summaries, &[])
        }
        MirStmtKind::PrepareFixedAssignment(phase) => {
            let mutation = &phase.assignment.mutation;
            let (mut effects, mut capabilities) =
                crate::statement_declared_requirements(&statement.kind);
            if let Some(crate::MirCallArgumentRegion {
                argument: crate::MirCallArg::Expansion(source),
                ..
            }) = phase.selector_region()
            {
                let (nested_effects, nested_capabilities) =
                    expansion_source_contract(source, summaries, &[]);
                effects.0.extend(nested_effects.0);
                capabilities.0.extend(nested_capabilities.0);
            }
            if matches!(
                phase.phase,
                crate::MirFixedPreparationPhase::FinishStep { .. }
            ) && (mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::Overloaded
                || mutation.shape_policy == runmat_hir::AssignmentShapePolicy::Overloaded
                || mutation.kind == runmat_hir::PlaceMutationKind::MemberAssign)
            {
                effects.0.insert(EffectKind::Unknown);
            }
            (effects, capabilities)
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
        MirStmtKind::StatementOutput(_) => crate::statement_declared_requirements(&statement.kind),
        MirStmtKind::WorkspaceEffect { .. } => {
            crate::statement_declared_requirements(&statement.kind)
        }
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

/// Complete conservative contract used for structurally present operations
/// whose flow entry is unreachable. It runs the same embedded-expression
/// walker as reachable inference, with no invented facts or literals.
pub(crate) fn conservative_rvalue_contract(
    value: &MirRvalue,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> (EffectSet, CapabilitySet) {
    rvalue_contract_with_facts_and_literals(value, summaries, &[], &BTreeMap::new(), None)
}

pub(crate) fn rvalue_contract_with_facts(
    value: &MirRvalue,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
) -> (EffectSet, CapabilitySet) {
    rvalue_contract_with_facts_and_literals(value, summaries, facts, &BTreeMap::new(), None)
}

fn rvalue_contract_with_facts_and_literals(
    value: &MirRvalue,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
    sequences: &BTreeMap<crate::MirSequenceLocalId, runmat_types::ValueSequenceFact>,
    proven_literals: Option<&[LiteralValue]>,
) -> (EffectSet, CapabilitySet) {
    let mut effects = EffectSet::default();
    let mut capabilities = CapabilitySet::default();
    match value {
        MirRvalue::Call(call) => {
            let unknown_literals = vec![LiteralValue::Unknown; call.args.len()];
            let literals = proven_literals.unwrap_or(&unknown_literals);
            let inference = infer_mir_call(
                call,
                facts,
                sequences,
                literals,
                summaries,
                OutputSelection::new(call.requested_outputs),
            );
            // `infer_mir_call` is the canonical operation-local inference
            // owner. A typed non-legacy catalog identity may soundly refine
            // its coarse declaration from facts and literals (for example,
            // TCP-only `close`). Other inference paths do not certify that
            // their effect result is complete.
            effects = inference.effects;
            capabilities = inference.capabilities;
            if !typed_catalog_contract_is_complete(call) {
                // Legacy, dynamic, method, and unresolved inference may
                // recover useful output facts without owning a complete
                // effect contract. Preserve the lowering declaration for
                // those paths. Only a typed non-legacy catalog identity is
                // allowed to refine that declaration away.
                let (declared_effects, declared_capabilities) =
                    crate::rvalue_declared_requirements(value);
                effects.0.extend(declared_effects.0);
                capabilities.0.extend(declared_capabilities.0);
            }
            if matches!(
                call.workspace_effect,
                Some(
                    runmat_builtins::BuiltinWorkspaceEffect::DynamicEval
                        | runmat_builtins::BuiltinWorkspaceEffect::DynamicEvalCurrentWorkspace
                )
            ) {
                capabilities.0.insert(CapabilityRequirement::HostRuntime);
            }
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
        MirRvalue::Future { .. } | MirRvalue::ExternalFuture { .. } | MirRvalue::Spawn(_) => {
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

fn typed_catalog_contract_is_complete(call: &crate::MirCall) -> bool {
    let crate::MirCallee::Static(runmat_hir::CallableIdentity::Builtin(identity)) = &call.callee
    else {
        return false;
    };
    runmat_builtins::builtin_catalog_entry_by_name(&identity.0).is_some_and(|entry| {
        !matches!(
            entry.contract.maturity,
            runmat_builtins::BuiltinContractMaturity::LegacyResolver
        )
    })
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

pub(crate) fn statement_contract_with_facts(
    statement: &MirStmt,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
) -> (EffectSet, CapabilitySet) {
    let (mut effects, mut capabilities) = statement_phase_contract(statement, summaries, facts);
    if let Some(value) = crate::statement_rvalue(&statement.kind) {
        let (nested_effects, nested_capabilities) =
            rvalue_contract_with_facts(value, summaries, facts);
        effects.0.extend(nested_effects.0);
        capabilities.0.extend(nested_capabilities.0);
    }
    (effects, capabilities)
}

/// Requirements executed by the store/statement phase, independently of its
/// preceding RHS operation. Contextual LHS selectors belong here, while the
/// cloned PlaceMutation annotation must not count as a second evaluation.
pub(crate) fn statement_phase_contract(
    statement: &MirStmt,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    facts: &[Option<ValueFact>],
) -> (EffectSet, CapabilitySet) {
    if let MirStmtKind::CaptureSequence { source, .. } = &statement.kind {
        return expansion_source_contract(source, summaries, facts);
    }
    let (mut effects, mut capabilities) =
        if let MirStmtKind::PreparedAssign { preparation, .. } = &statement.kind {
            // The retained target's protocol write executes here; preparation
            // owns selector expressions but cannot license the later callback.
            let mutation = &preparation.mutation;
            if mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::Overloaded
                || mutation.shape_policy == runmat_hir::AssignmentShapePolicy::Overloaded
                || mutation.kind == runmat_hir::PlaceMutationKind::MemberAssign
            {
                (
                    EffectSet([EffectKind::Unknown].into_iter().collect()),
                    CapabilitySet::default(),
                )
            } else {
                Default::default()
            }
        } else if crate::statement_rvalue(&statement.kind).is_some() {
            Default::default()
        } else {
            statement_contract(statement, summaries)
        };
    statement
        .kind
        .visit_statement_expression_steps_dyn(&mut |steps| {
            let (nested_effects, nested_capabilities) =
                expression_steps_contract(steps, summaries, facts);
            effects.0.extend(nested_effects.0);
            capabilities.0.extend(nested_capabilities.0);
        });
    (effects, capabilities)
}

#[cfg(test)]
mod literal_capability_tests {
    use super::*;

    fn call_and_state(source: &str) -> (crate::MirCall, FlowState) {
        let ast = runmat_parser::parse(source).expect("parse capability fixture");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower capability fixture HIR");
        let mir =
            crate::lowering::lower_assembly(&hir.assembly).expect("lower capability fixture MIR");
        let body = mir.bodies.values().next().expect("fixture body");
        let call = body
            .blocks
            .iter()
            .flat_map(|block| &block.statements)
            .find_map(|statement| match &statement.kind {
                MirStmtKind::Assign {
                    value: MirRvalue::Call(call),
                    ..
                }
                | MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call.clone()),
                _ => None,
            })
            .expect("fixture call");
        (call, FlowState::entry(body, &[], &[], None))
    }

    #[test]
    fn call_capabilities_see_only_proven_source_literals() {
        let (local, state) = call_and_state("out = imread('image.png');");
        assert_eq!(
            proven_call_literals(&local, &state),
            vec![LiteralValue::Character("image.png".into())]
        );
        let (remote, state) = call_and_state("out = imread('https://example.test/a.png');");
        assert_eq!(
            proven_call_literals(&remote, &state),
            vec![LiteralValue::Character("https://example.test/a.png".into())]
        );
        let (dynamic, mut state) =
            call_and_state("function out = load_image(path)\n out = imread(path);\nend");
        assert_eq!(
            proven_call_literals(&dynamic, &state),
            vec![LiteralValue::Unknown]
        );
        let Some(crate::MirCallArg::Single(crate::MirOperand::Local(local))) = dynamic.args.first()
        else {
            panic!("dynamic source should be a local argument");
        };
        state.locals[local.0].set(
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            LiteralValue::Character("file:///images/a.png".into()),
        );
        assert_eq!(
            proven_call_literals(&dynamic, &state),
            vec![LiteralValue::Character("file:///images/a.png".into())]
        );
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
