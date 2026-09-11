use std::collections::{BTreeMap, VecDeque};

use runmat_hir::{FunctionId, Span};
use runmat_types::{
    CapabilityRequirement, DynamicReason, EffectKind, LiteralValue, ProgramFunctionId,
    ProgramPointId, ProgramSpan, RegionValueId, ValueFact, ValueKindFact,
};

use crate::{
    BasicBlock, BasicBlockId, MirBody, MirOutputTarget, MirRvalue, MirStmt, MirStmtKind,
    MirTerminatorKind,
};

use super::state::FlowState;
use crate::analysis::inference::FunctionSummary;
use crate::analysis::inference::{
    append_inference_diagnostics, apply_rvalue_contract, assign_place, infer_expansion_sequence,
    infer_rvalue, infer_rvalue_outputs, operand_fact, rvalue_literal,
    statement_contract_with_facts,
};
use crate::analysis::{AssignmentFact, ProgramLocalFact, ProgramPointFacts};

#[path = "flow/call_observations.rs"]
mod call_observations;
pub(crate) use call_observations::CallObservation;

const WIDEN_AFTER_UPDATE: usize = 4;
const MAX_BLOCK_UPDATES: usize = 64;

pub(crate) struct BodyFlow {
    pub points: Vec<ProgramPointFacts>,
    pub final_state: FlowState,
    pub return_facts: Vec<Vec<ValueFact>>,
    pub widened: bool,
    pub converged: bool,
    pub diagnostics: Vec<crate::MirDiagnostic>,
    pub calls: Vec<CallObservation>,
}

pub(crate) fn analyze_body(
    body: &MirBody,
    function: ProgramFunctionId,
    parameters: &[ValueFact],
    captures: &[ValueFact],
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> BodyFlow {
    if body.blocks.is_empty() {
        return BodyFlow {
            points: Vec::new(),
            final_state: FlowState::entry(body, parameters, captures),
            return_facts: Vec::new(),
            widened: false,
            converged: true,
            diagnostics: Vec::new(),
            calls: Vec::new(),
        };
    }

    let block_index: BTreeMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.id, index))
        .collect();
    let mut entries = vec![None; body.blocks.len()];
    let mut updates = vec![0usize; body.blocks.len()];
    entries[0] = Some(FlowState::entry(body, parameters, captures));
    let mut worklist = VecDeque::from([0usize]);
    let mut widened = false;
    let mut converged = true;

    while let Some(block_index_value) = worklist.pop_front() {
        let Some(input) = entries[block_index_value].clone() else {
            continue;
        };
        let block = &body.blocks[block_index_value];
        let mut state = input;
        transfer_statements(&block.statements, &mut state, summaries, &mut Vec::new());
        for (successor, outgoing) in edge_states(&block.terminator.kind, &state, summaries) {
            let Some(&successor_index) = block_index.get(&successor) else {
                continue;
            };
            updates[successor_index] += 1;
            if updates[successor_index] > MAX_BLOCK_UPDATES {
                converged = false;
                continue;
            }
            let should_widen = updates[successor_index] > WIDEN_AFTER_UPDATE;
            widened |= should_widen;
            let changed = match &mut entries[successor_index] {
                Some(existing) => existing.join_from(&outgoing, should_widen),
                slot @ None => {
                    *slot = Some(outgoing);
                    true
                }
            };
            if changed && !worklist.contains(&successor_index) {
                worklist.push_back(successor_index);
            }
        }
    }

    let mut points = Vec::new();
    let mut final_state = FlowState::entry(body, parameters, captures);
    let mut has_final = false;
    let mut return_facts = Vec::new();
    let mut diagnostics = Vec::new();
    let mut calls = Vec::new();
    for (block_index_value, block) in body.blocks.iter().enumerate() {
        let Some(mut state) = entries[block_index_value].clone() else {
            continue;
        };
        push_point(
            &mut points,
            function,
            block,
            0,
            block_entry_span(block),
            &state,
        );
        let mut pending_mutation = None;
        for (statement_index, statement) in block.statements.iter().enumerate() {
            call_observations::collect(statement, &state, summaries, &mut calls);
            transfer_statement(
                statement,
                &mut state,
                &mut pending_mutation,
                summaries,
                &mut diagnostics,
            );
            push_point(
                &mut points,
                function,
                block,
                statement_index + 1,
                statement.span,
                &state,
            );
        }
        if let MirTerminatorKind::Return(outputs) = &block.terminator.kind {
            return_facts.push(
                outputs
                    .iter()
                    .map(|operand| operand_fact(operand, &state))
                    .collect(),
            );
        }
        if !has_final {
            final_state = state;
            has_final = true;
        } else {
            final_state.join_from(&state, false);
        }
    }
    points.sort_by_key(|point| point.point);

    BodyFlow {
        points,
        final_state,
        return_facts,
        widened,
        converged,
        diagnostics,
        calls,
    }
}

fn transfer_statements(
    statements: &[MirStmt],
    state: &mut FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    diagnostics: &mut Vec<crate::MirDiagnostic>,
) {
    let mut pending_mutation = None;
    for statement in statements {
        transfer_statement(
            statement,
            state,
            &mut pending_mutation,
            summaries,
            diagnostics,
        );
    }
}

fn transfer_statement(
    statement: &MirStmt,
    state: &mut FlowState,
    pending_mutation: &mut Option<crate::MirPlaceMutation>,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    diagnostics: &mut Vec<crate::MirDiagnostic>,
) {
    match &statement.kind {
        MirStmtKind::Assign { place, value } => {
            transfer_short_circuit_temps(value, state, summaries, diagnostics);
            let assigned = infer_rvalue(value, state, summaries, statement.span, diagnostics);
            let literal = rvalue_literal(value, state);
            assign_place(
                place,
                assigned,
                literal,
                pending_mutation.take().as_ref(),
                state,
            );
            apply_rvalue_contract(value, state, summaries);
        }
        MirStmtKind::MultiAssign { targets, value } => {
            transfer_short_circuit_temps(value, state, summaries, diagnostics);
            let outputs = infer_rvalue_outputs(
                value,
                state,
                summaries,
                Some(targets),
                statement.span,
                diagnostics,
            );
            for (index, target) in targets.targets.iter().enumerate() {
                match target {
                    MirOutputTarget::Place(place) => assign_place(
                        place,
                        outputs.get(index).cloned().unwrap_or_else(dynamic_value),
                        LiteralValue::Unknown,
                        pending_mutation.take().as_ref(),
                        state,
                    ),
                    MirOutputTarget::Sequence(target) => {
                        if let Some(local) = place_root_local(target.base()) {
                            if let Some(fact) = state.locals.get_mut(local.0) {
                                fact.set(
                                    ValueFact::unknown(DynamicReason::RuntimeValue),
                                    LiteralValue::Unknown,
                                );
                            }
                        }
                    }
                    MirOutputTarget::Discard => {}
                }
            }
            apply_rvalue_contract(value, state, summaries);
        }
        MirStmtKind::SequenceAssign { target, value } => {
            transfer_short_circuit_temps(value, state, summaries, diagnostics);
            let _ = infer_rvalue(value, state, summaries, statement.span, diagnostics);
            let base = match target {
                crate::MirSequenceTarget::Member { base, .. }
                | crate::MirSequenceTarget::DynamicMember { base, .. }
                | crate::MirSequenceTarget::CellContents { base, .. } => base,
            };
            if let Some(local) = place_root_local(base) {
                if let Some(fact) = state.locals.get_mut(local.0) {
                    fact.set(
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                        LiteralValue::Unknown,
                    );
                }
            }
            apply_rvalue_contract(value, state, summaries);
        }
        MirStmtKind::CaptureSequence {
            destination,
            source,
        } => {
            source.visit_outer_operands(|operand| {
                let _ = operand_fact(operand, state);
            });
            let inferred = infer_expansion_sequence(source, state, summaries);
            append_inference_diagnostics(
                &inferred.diagnostics,
                statement.span,
                "sequence-inference",
                diagnostics,
            );
            state.set_sequence(*destination, inferred.sequence);
            let facts = state.value_facts();
            let (effects, capabilities) =
                statement_contract_with_facts(statement, summaries, &facts);
            state.effects.0.extend(effects.0);
            state.capabilities.0.extend(capabilities.0);
        }
        MirStmtKind::Expr(value) => {
            transfer_short_circuit_temps(value, state, summaries, diagnostics);
            let _ = infer_rvalue(value, state, summaries, statement.span, diagnostics);
            apply_rvalue_contract(value, state, summaries);
        }
        MirStmtKind::PlaceMutation(mutation) => *pending_mutation = Some(mutation.clone()),
        MirStmtKind::WorkspaceEffect { bindings, .. } => {
            state.effects.0.insert(EffectKind::WorkspaceWrite);
            for binding in bindings {
                if let Some(local) = state.locals.get_mut(binding.0) {
                    local.set(
                        ValueFact::unknown(DynamicReason::RuntimeValue),
                        LiteralValue::Unknown,
                    );
                }
            }
        }
        MirStmtKind::EnvironmentEffect(_) => {
            state.effects.0.insert(EffectKind::EnvironmentWrite);
            for local in &mut state.locals {
                if let Some(fact) = &mut local.fact {
                    fact.invalidation
                        .0
                        .insert(runmat_types::InvalidationCause::RuntimePolicyChanged);
                }
            }
        }
    }
}

fn place_root_local(place: &crate::MirPlace) -> Option<crate::MirLocalId> {
    match place {
        crate::MirPlace::Local(local) => Some(*local),
        crate::MirPlace::Binding(_) => None,
        crate::MirPlace::Member(base, _)
        | crate::MirPlace::DynamicMember(base, _)
        | crate::MirPlace::Index(base, _) => place_root_local(base),
    }
}

fn transfer_short_circuit_temps(
    value: &MirRvalue,
    state: &mut FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    diagnostics: &mut Vec<crate::MirDiagnostic>,
) {
    let MirRvalue::ShortCircuit { right_temps, .. } = value else {
        return;
    };
    let mut executed = state.clone();
    transfer_statements(right_temps, &mut executed, summaries, diagnostics);
    state.join_from(&executed, false);
}

fn edge_states(
    kind: &MirTerminatorKind,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
) -> Vec<(BasicBlockId, FlowState)> {
    match kind {
        MirTerminatorKind::Goto(target) => vec![(*target, state.clone())],
        MirTerminatorKind::Branch {
            then_block,
            else_block,
            ..
        } => vec![(*then_block, state.clone()), (*else_block, state.clone())],
        MirTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, target)| (*target, state.clone()))
            .chain(std::iter::once((*otherwise, state.clone())))
            .collect(),
        MirTerminatorKind::For {
            binding,
            iterable,
            body_block,
            exit_block,
        } => loop_edges(
            *binding,
            iterable,
            *body_block,
            *exit_block,
            state,
            summaries,
            false,
        ),
        MirTerminatorKind::ParFor {
            binding,
            iterable,
            body_block,
            exit_block,
            ..
        } => loop_edges(
            *binding,
            iterable,
            *body_block,
            *exit_block,
            state,
            summaries,
            true,
        ),
        MirTerminatorKind::Spmd {
            body_block,
            exit_block,
            ..
        } => {
            let mut body = state.clone();
            body.capabilities
                .0
                .insert(CapabilityRequirement::ParallelRuntime);
            vec![(*body_block, body), (*exit_block, state.clone())]
        }
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            catch_binding,
        } => {
            let mut caught = state.clone();
            if let Some(local) = catch_binding {
                caught.locals[local.0].set(
                    ValueFact::scalar(ValueKindFact::Exception(runmat_types::ExceptionFact {
                        identifier: None,
                    })),
                    LiteralValue::Unknown,
                );
            }
            vec![(*try_block, state.clone()), (*catch_block, caught)]
        }
        MirTerminatorKind::Await {
            future,
            result,
            resume,
        } => {
            let output = match operand_fact(future, state).kind {
                ValueKindFact::Execution(runmat_types::ExecutionFact::Future {
                    output, ..
                })
                | ValueKindFact::Execution(runmat_types::ExecutionFact::Task { output, .. }) => {
                    output.first_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
                }
                _ => dynamic_value(),
            };
            let mut resumed = state.clone();
            resumed.effects.0.insert(EffectKind::MaySuspend);
            resumed
                .capabilities
                .0
                .insert(CapabilityRequirement::ParallelRuntime);
            if let Some(place) = result {
                assign_place(place, output, LiteralValue::Unknown, None, &mut resumed);
            }
            vec![(*resume, resumed)]
        }
        MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => Vec::new(),
    }
}

fn loop_edges(
    binding: crate::MirLocalId,
    iterable: &MirRvalue,
    body_block: BasicBlockId,
    exit_block: BasicBlockId,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    parallel: bool,
) -> Vec<(BasicBlockId, FlowState)> {
    let mut body = state.clone();
    let iterable = infer_rvalue(
        iterable,
        &mut body,
        summaries,
        Span::default(),
        &mut Vec::new(),
    );
    let element = iteration_element(&iterable);
    body.locals[binding.0].set(element, LiteralValue::Unknown);
    if parallel {
        body.capabilities
            .0
            .insert(CapabilityRequirement::ParallelRuntime);
    }
    vec![(body_block, body), (exit_block, state.clone())]
}

fn iteration_element(iterable: &ValueFact) -> ValueFact {
    let mut fact = iterable.clone();
    fact.shape = runmat_types::ShapeFact::Scalar;
    fact.storage = runmat_types::StorageFact::Scalar;
    fact
}

fn push_point(
    points: &mut Vec<ProgramPointFacts>,
    function: ProgramFunctionId,
    block: &BasicBlock,
    position: usize,
    span: Span,
    state: &FlowState,
) {
    let Ok(block) = u32::try_from(block.id.0) else {
        return;
    };
    let Ok(position) = u32::try_from(position) else {
        return;
    };
    let locals = state
        .final_facts()
        .filter_map(|(local, state)| {
            Some(ProgramLocalFact {
                value: RegionValueId {
                    function,
                    local: u32::try_from(local).ok()?,
                },
                assignment: match state.assignment {
                    super::super::InitFact::Unassigned => AssignmentFact::Unassigned,
                    super::super::InitFact::MaybeAssigned => AssignmentFact::MaybeAssigned,
                    super::super::InitFact::DefinitelyAssigned => {
                        AssignmentFact::DefinitelyAssigned
                    }
                },
                fact: state.fact.clone(),
            })
        })
        .collect();
    points.push(ProgramPointFacts {
        point: ProgramPointId {
            function,
            block,
            position,
        },
        span: ProgramSpan {
            start: span.start as u64,
            end: span.end as u64,
        },
        locals,
        effects: state.effects.clone(),
        capabilities: state.capabilities.clone(),
    });
}

fn block_entry_span(block: &BasicBlock) -> Span {
    block
        .statements
        .first()
        .map_or(block.terminator.span, |statement| statement.span)
}

fn dynamic_value() -> ValueFact {
    ValueFact::unknown(DynamicReason::Unspecified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_calls_do_not_hide_static_calls_in_contextual_arguments() {
        let ast = runmat_parser::parse(
            r#"
function out = main(callable, values, flag)
  out = callable(values{[flag && helper(1 + 2), end]});
end
function out = helper(index)
  out = index;
end
"#,
        )
        .expect("parse fixture");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower HIR fixture");
        let mir = crate::lowering::lower_assembly(&hir.assembly).expect("lower MIR fixture");
        let program = super::super::interprocedural::analyze_program(&mir);
        let by_name = mir
            .functions
            .iter()
            .map(|(id, metadata)| (metadata.name.0.as_str(), *id))
            .collect::<BTreeMap<_, _>>();
        let main = by_name["main"];
        let flow = analyze_body(
            &mir.bodies[&main],
            ProgramFunctionId(u32::try_from(main.0).unwrap()),
            &program.parameters[&main],
            &program.captures[&main],
            &program.summaries,
        );
        let helper = by_name["helper"];
        let observations = flow
            .calls
            .iter()
            .filter(|observation| observation.callee == helper)
            .collect::<Vec<_>>();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].arguments.len(), 1);
        assert!(matches!(
            observations[0].arguments[0].kind,
            ValueKindFact::Numeric(_)
        ));
        assert_eq!(
            observations[0].arguments[0].shape.known_dims(),
            Some(vec![Some(1), Some(1)])
        );
    }

    #[test]
    fn ordinary_short_circuit_calls_are_observed_once() {
        let ast = runmat_parser::parse(
            r#"
function out = main(flag)
  out = flag && helper(1 + 2);
end
function out = helper(value)
  out = value;
end
"#,
        )
        .expect("parse fixture");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower HIR fixture");
        let mir = crate::lowering::lower_assembly(&hir.assembly).expect("lower MIR fixture");
        let program = super::super::interprocedural::analyze_program(&mir);
        let by_name = mir
            .functions
            .iter()
            .map(|(id, metadata)| (metadata.name.0.as_str(), *id))
            .collect::<BTreeMap<_, _>>();
        let main = by_name["main"];
        let flow = analyze_body(
            &mir.bodies[&main],
            ProgramFunctionId(u32::try_from(main.0).unwrap()),
            &program.parameters[&main],
            &program.captures[&main],
            &program.summaries,
        );
        let helper = by_name["helper"];
        assert_eq!(
            flow.calls
                .iter()
                .filter(|observation| observation.callee == helper)
                .count(),
            1
        );
    }
}
