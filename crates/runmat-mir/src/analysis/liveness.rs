use crate::{
    BasicBlockId, MirBody, MirIndexComponent, MirIndexing, MirLocalId, MirOperand, MirPlace,
    MirRvalue, MirStmtKind, MirTerminatorKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LivenessFacts {
    pub live_across_await: Vec<(BasicBlockId, Vec<MirLocalId>)>,
}

pub fn analyze_liveness(body: &MirBody) -> LivenessFacts {
    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut facts = LivenessFacts::default();

    for block in &body.blocks {
        if let MirTerminatorKind::Await { future, resume, .. } = &block.terminator.kind {
            let mut live = HashSet::new();
            collect_operand_read(future, &mut live);
            collect_reachable_reads(*resume, body, &block_index, &mut live);
            let mut live: Vec<_> = live.into_iter().collect();
            live.sort_by_key(|local| local.0);
            facts.live_across_await.push((block.id, live));
        }
    }

    facts
}

fn collect_reachable_reads(
    start: BasicBlockId,
    body: &MirBody,
    block_index: &HashMap<BasicBlockId, usize>,
    live: &mut HashSet<MirLocalId>,
) {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([start]);
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        let Some(&idx) = block_index.get(&id) else {
            continue;
        };
        let block = &body.blocks[idx];
        for stmt in &block.statements {
            match &stmt.kind {
                MirStmtKind::Assign { place, value } => {
                    collect_rvalue_reads(value, live);
                    collect_place_reads(place, live);
                }
                MirStmtKind::MultiAssign { targets, value } => {
                    collect_rvalue_reads(value, live);
                    for target in &targets.targets {
                        if let crate::MirOutputTarget::Place(place) = target {
                            collect_place_reads(place, live);
                        }
                    }
                }
                MirStmtKind::Expr(value) => collect_rvalue_reads(value, live),
                MirStmtKind::PlaceMutation(_) | MirStmtKind::WorkspaceEffect { .. } => {}
            }
        }
        collect_terminator_reads(&block.terminator.kind, live);
        for successor in successors(&block.terminator.kind) {
            queue.push_back(successor);
        }
    }
}

fn collect_terminator_reads(kind: &MirTerminatorKind, live: &mut HashSet<MirLocalId>) {
    match kind {
        MirTerminatorKind::Branch { cond, .. } => collect_operand_read(cond, live),
        MirTerminatorKind::For { iterable, .. } => collect_rvalue_reads(iterable, live),
        MirTerminatorKind::Return(outputs) => {
            for output in outputs {
                collect_operand_read(output, live);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            collect_operand_read(future, live);
            if let Some(place) = result {
                collect_place_reads(place, live);
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
}

fn collect_rvalue_reads(value: &MirRvalue, live: &mut HashSet<MirLocalId>) {
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) | MirRvalue::Spawn(operand) => {
            collect_operand_read(operand, live);
        }
        MirRvalue::Binary(left, _, right) => {
            collect_operand_read(left, live);
            collect_operand_read(right, live);
        }
        MirRvalue::Range { start, step, end } => {
            collect_operand_read(start, live);
            if let Some(step) = step {
                collect_operand_read(step, live);
            }
            collect_operand_read(end, live);
        }
        MirRvalue::Call(call) => {
            for arg in &call.args {
                collect_operand_read(arg, live);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                collect_operand_read(element, live);
            }
        }
        MirRvalue::Index { base, indexing } => {
            collect_operand_read(base, live);
            collect_indexing_reads(indexing, live);
        }
        MirRvalue::Future(_) => {}
    }
}

fn collect_place_reads(place: &MirPlace, live: &mut HashSet<MirLocalId>) {
    match place {
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
        MirPlace::Member(base, _) => collect_place_reads(base, live),
        MirPlace::Index(base, indexing) => {
            collect_place_reads(base, live);
            collect_indexing_reads(indexing, live);
        }
    }
}

fn collect_indexing_reads(indexing: &MirIndexing, live: &mut HashSet<MirLocalId>) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) | MirIndexComponent::Logical(operand) => {
                collect_operand_read(operand, live);
            }
            MirIndexComponent::Colon | MirIndexComponent::End { .. } => {}
        }
    }
}

fn collect_operand_read(operand: &MirOperand, live: &mut HashSet<MirLocalId>) {
    if let MirOperand::Local(local) = operand {
        live.insert(*local);
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
