use crate::{
    BasicBlock, BasicBlockId, MirBody, MirDiagnostic, MirDiagnosticSeverity, MirIndexComponent,
    MirIndexing, MirLocalId, MirLocalKind, MirOperand, MirPlace, MirRvalue, MirStmtKind,
    MirTerminatorKind,
};
use runmat_hir::{ShapeFact, Span, TypeFact, ValueFlowFact};
use std::collections::{HashMap, VecDeque};

use super::{AnalysisStore, InitFact, MirLocalFact};

#[derive(Debug, Clone)]
struct InitDataflowResult {
    in_states: Vec<Option<Vec<InitFact>>>,
    final_state: Vec<InitFact>,
}

pub fn analyze_body(body: &MirBody, store: &mut AnalysisStore) {
    let result = compute_init_dataflow(body);

    for local in &body.locals {
        store.mir_locals.insert(
            local.id,
            MirLocalFact {
                ty: TypeFact::Unknown,
                shape: ShapeFact::Unknown,
                value_flow: ValueFlowFact::UnknownList,
                initialized: result.final_state[local.id.0],
            },
        );
    }
}

pub fn diagnose_uninitialized_reads(body: &MirBody) -> Vec<MirDiagnostic> {
    let result = compute_init_dataflow(body);
    let mut diagnostics = Vec::new();

    for (idx, block) in body.blocks.iter().enumerate() {
        let Some(mut state) = result.in_states[idx].clone() else {
            continue;
        };
        diagnose_block(block, &mut state, &mut diagnostics);
    }

    diagnostics
}

fn compute_init_dataflow(body: &MirBody) -> InitDataflowResult {
    let local_count = body.locals.len();
    if body.blocks.is_empty() {
        return InitDataflowResult {
            in_states: Vec::new(),
            final_state: Vec::new(),
        };
    }

    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();

    let mut initial = vec![InitFact::Unassigned; local_count];
    for local in &body.locals {
        if matches!(local.kind, MirLocalKind::Parameter | MirLocalKind::Capture) {
            initial[local.id.0] = InitFact::DefinitelyAssigned;
        }
    }

    let mut in_states: Vec<Option<Vec<InitFact>>> = vec![None; body.blocks.len()];
    let mut out_states: Vec<Vec<InitFact>> =
        vec![vec![InitFact::Unassigned; local_count]; body.blocks.len()];
    in_states[0] = Some(initial);

    let mut worklist = VecDeque::from([0usize]);
    while let Some(block_idx) = worklist.pop_front() {
        let Some(input) = in_states[block_idx].clone() else {
            continue;
        };
        let block = &body.blocks[block_idx];
        let output = transfer_block(block, input);
        out_states[block_idx] = output.clone();

        for successor in successors(&block.terminator.kind) {
            let Some(&successor_idx) = block_index.get(&successor) else {
                continue;
            };
            let changed = match &mut in_states[successor_idx] {
                Some(existing) => join_into(existing, &output),
                slot @ None => {
                    *slot = Some(output.clone());
                    true
                }
            };
            if changed {
                worklist.push_back(successor_idx);
            }
        }
    }

    let mut final_state: Option<Vec<InitFact>> = None;
    for (idx, block) in body.blocks.iter().enumerate() {
        if matches!(block.terminator.kind, MirTerminatorKind::Return(_)) {
            match &mut final_state {
                Some(state) => {
                    join_into(state, &out_states[idx]);
                }
                None => final_state = Some(out_states[idx].clone()),
            }
        }
    }
    InitDataflowResult {
        in_states,
        final_state: final_state.unwrap_or_else(|| vec![InitFact::Unassigned; local_count]),
    }
}

fn transfer_block(block: &crate::BasicBlock, mut state: Vec<InitFact>) -> Vec<InitFact> {
    for stmt in &block.statements {
        match &stmt.kind {
            MirStmtKind::Assign { place, .. } => mark_place_assigned(place, &mut state),
            MirStmtKind::MultiAssign { targets, .. } => {
                for target in &targets.targets {
                    if let crate::MirOutputTarget::Place(place) = target {
                        mark_place_assigned(place, &mut state);
                    }
                }
            }
            MirStmtKind::Expr(_)
            | MirStmtKind::PlaceMutation(_)
            | MirStmtKind::WorkspaceEffect { .. } => {}
        }
    }

    match &block.terminator.kind {
        MirTerminatorKind::For { binding, .. } => {
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::Await { result, .. } => {
            if let Some(place) = result {
                mark_place_assigned(place, &mut state);
            }
        }
        _ => {}
    }
    state
}

fn diagnose_block(
    block: &BasicBlock,
    state: &mut [InitFact],
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for stmt in &block.statements {
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => {
                diagnose_rvalue_reads(value, state, stmt.span, diagnostics);
                diagnose_place_reads(place, state, stmt.span, diagnostics);
                mark_place_assigned(place, state);
            }
            MirStmtKind::MultiAssign { targets, value } => {
                diagnose_rvalue_reads(value, state, stmt.span, diagnostics);
                for target in &targets.targets {
                    if let crate::MirOutputTarget::Place(place) = target {
                        diagnose_place_reads(place, state, stmt.span, diagnostics);
                        mark_place_assigned(place, state);
                    }
                }
            }
            MirStmtKind::Expr(value) => diagnose_rvalue_reads(value, state, stmt.span, diagnostics),
            MirStmtKind::PlaceMutation(_) | MirStmtKind::WorkspaceEffect { .. } => {}
        }
    }

    match &block.terminator.kind {
        MirTerminatorKind::Branch { cond, .. } => {
            diagnose_operand_read(cond, state, block.terminator.span, diagnostics);
        }
        MirTerminatorKind::Switch { discr, cases, .. } => {
            diagnose_operand_read(discr, state, block.terminator.span, diagnostics);
            for (case, _) in cases {
                diagnose_operand_read(case, state, block.terminator.span, diagnostics);
            }
        }
        MirTerminatorKind::For {
            binding, iterable, ..
        } => {
            diagnose_rvalue_reads(iterable, state, block.terminator.span, diagnostics);
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::Return(outputs) => {
            for output in outputs {
                diagnose_operand_read(output, state, block.terminator.span, diagnostics);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            diagnose_operand_read(future, state, block.terminator.span, diagnostics);
            if let Some(place) = result {
                diagnose_place_reads(place, state, block.terminator.span, diagnostics);
                mark_place_assigned(place, state);
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
}

fn diagnose_rvalue_reads(
    value: &MirRvalue,
    state: &[InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) | MirRvalue::Spawn(operand) => {
            diagnose_operand_read(operand, state, span, diagnostics);
        }
        MirRvalue::Binary(left, _, right) => {
            diagnose_operand_read(left, state, span, diagnostics);
            diagnose_operand_read(right, state, span, diagnostics);
        }
        MirRvalue::Range { start, step, end } => {
            diagnose_operand_read(start, state, span, diagnostics);
            if let Some(step) = step {
                diagnose_operand_read(step, state, span, diagnostics);
            }
            diagnose_operand_read(end, state, span, diagnostics);
        }
        MirRvalue::Call(call) => {
            for arg in &call.args {
                diagnose_operand_read(arg, state, span, diagnostics);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                diagnose_operand_read(element, state, span, diagnostics);
            }
        }
        MirRvalue::Index { base, indexing } => {
            diagnose_operand_read(base, state, span, diagnostics);
            diagnose_indexing_reads(indexing, state, span, diagnostics);
        }
        MirRvalue::Future(_) => {}
    }
}

fn diagnose_place_reads(
    place: &MirPlace,
    state: &[InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match place {
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
        MirPlace::Member(base, _) => diagnose_place_reads(base, state, span, diagnostics),
        MirPlace::Index(base, indexing) => {
            diagnose_place_reads(base, state, span, diagnostics);
            diagnose_indexing_reads(indexing, state, span, diagnostics);
        }
    }
}

fn diagnose_indexing_reads(
    indexing: &MirIndexing,
    state: &[InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) | MirIndexComponent::Logical(operand) => {
                diagnose_operand_read(operand, state, span, diagnostics);
            }
            MirIndexComponent::Colon | MirIndexComponent::End { .. } => {}
        }
    }
}

fn diagnose_operand_read(
    operand: &MirOperand,
    state: &[InitFact],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    let MirOperand::Local(local) = operand else {
        return;
    };
    match state[local.0] {
        InitFact::DefinitelyAssigned => {}
        InitFact::Unassigned => diagnostics.push(init_diagnostic(
            "RM-MIR0001",
            "local may be read before it is assigned",
            "this local is read before any assignment reaches this point",
            span,
        )),
        InitFact::MaybeAssigned => diagnostics.push(init_diagnostic(
            "RM-MIR0002",
            "local may be read before assignment on some control-flow paths",
            "this local is not definitely assigned on every path reaching this point",
            span,
        )),
    }
}

fn init_diagnostic(
    code: &'static str,
    message: &'static str,
    label: &'static str,
    span: Span,
) -> MirDiagnostic {
    MirDiagnostic::new(code, MirDiagnosticSeverity::Error, message, span)
        .with_primary_label(label)
        .with_category("definite-assignment")
}

fn mark_place_assigned(place: &MirPlace, state: &mut [InitFact]) {
    if let Some(local) = place_root(place) {
        state[local.0] = InitFact::DefinitelyAssigned;
    }
}

fn place_root(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Member(base, _) | MirPlace::Index(base, _) => place_root(base),
        MirPlace::Binding(_) => None,
    }
}

fn successors(kind: &MirTerminatorKind) -> Vec<BasicBlockId> {
    match kind {
        MirTerminatorKind::Goto(target) => vec![*target],
        MirTerminatorKind::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        MirTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, block)| *block)
            .chain(std::iter::once(*otherwise))
            .collect(),
        MirTerminatorKind::For {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await {
            resume, cleanup, ..
        } => cleanup.map_or_else(|| vec![*resume], |cleanup| vec![*resume, cleanup]),
        MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => Vec::new(),
    }
}

fn join_into(existing: &mut [InitFact], incoming: &[InitFact]) -> bool {
    let mut changed = false;
    for (slot, incoming) in existing.iter_mut().zip(incoming) {
        let joined = join_init(*slot, *incoming);
        if *slot != joined {
            *slot = joined;
            changed = true;
        }
    }
    changed
}

fn join_init(left: InitFact, right: InitFact) -> InitFact {
    match (left, right) {
        (InitFact::Unassigned, InitFact::Unassigned) => InitFact::Unassigned,
        (InitFact::DefinitelyAssigned, InitFact::DefinitelyAssigned) => {
            InitFact::DefinitelyAssigned
        }
        _ => InitFact::MaybeAssigned,
    }
}
