use crate::{
    BasicBlockId, MirBody, MirIndexComponent, MirIndexing, MirLocalId, MirOperand, MirPlace,
    MirRvalue, MirStmtKind, MirTerminatorKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    let mut block_use_def = Vec::new();
    for block in &body.blocks {
        block_use_def.push(block_uses_defs(block));
    }
    let mut live_in = vec![HashSet::<MirLocalId>::new(); body.blocks.len()];
    let mut live_out = vec![HashSet::<MirLocalId>::new(); body.blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (idx, block) in body.blocks.iter().enumerate().rev() {
            let mut out = HashSet::new();
            for successor in successors(&block.terminator.kind) {
                if let Some(&successor_idx) = block_index.get(&successor) {
                    out.extend(live_in[successor_idx].iter().copied());
                }
            }
            let (uses, defs) = &block_use_def[idx];
            let mut input = uses.clone();
            input.extend(out.difference(defs).copied());
            if out != live_out[idx] || input != live_in[idx] {
                live_out[idx] = out;
                live_in[idx] = input;
                changed = true;
            }
        }
    }
    let mut facts = LivenessFacts::default();

    for block in &body.blocks {
        if let MirTerminatorKind::Await {
            future,
            result,
            resume,
            ..
        } = &block.terminator.kind
        {
            let mut live = block_index
                .get(resume)
                .map(|idx| live_in[*idx].clone())
                .unwrap_or_default();
            if let Some(place) = result {
                if let Some(local) = place_root(place) {
                    live.remove(&local);
                }
            }
            collect_operand_read(future, &mut live);
            let mut live: Vec<_> = live.into_iter().collect();
            live.sort_by_key(|local| local.0);
            facts.live_across_await.push((block.id, live));
        }
    }

    facts
}

fn block_uses_defs(block: &crate::BasicBlock) -> (HashSet<MirLocalId>, HashSet<MirLocalId>) {
    let mut uses = HashSet::new();
    let mut defs = HashSet::new();
    for stmt in &block.statements {
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => {
                collect_rvalue_reads_with_defs(value, &mut uses, &defs);
                collect_place_reads_with_defs(place, &mut uses, &defs);
                if let Some(local) = place_root(place) {
                    defs.insert(local);
                }
            }
            MirStmtKind::MultiAssign { targets, value } => {
                collect_rvalue_reads_with_defs(value, &mut uses, &defs);
                for target in &targets.targets {
                    if let crate::MirOutputTarget::Place(place) = target {
                        collect_place_reads_with_defs(place, &mut uses, &defs);
                        if let Some(local) = place_root(place) {
                            defs.insert(local);
                        }
                    }
                }
            }
            MirStmtKind::Expr(value) => collect_rvalue_reads_with_defs(value, &mut uses, &defs),
            MirStmtKind::PlaceMutation(_)
            | MirStmtKind::WorkspaceEffect { .. }
            | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }
    collect_terminator_reads_with_defs(&block.terminator.kind, &mut uses, &defs);
    (uses, defs)
}

fn collect_terminator_reads_with_defs(
    kind: &MirTerminatorKind,
    uses: &mut HashSet<MirLocalId>,
    defs: &HashSet<MirLocalId>,
) {
    match kind {
        MirTerminatorKind::Branch { cond, .. } => collect_operand_read_with_defs(cond, uses, defs),
        MirTerminatorKind::Switch { discr, cases, .. } => {
            collect_operand_read_with_defs(discr, uses, defs);
            for (case, _) in cases {
                collect_operand_read_with_defs(case, uses, defs);
            }
        }
        MirTerminatorKind::For {
            binding, iterable, ..
        } => {
            collect_rvalue_reads_with_defs(iterable, uses, defs);
            if !defs.contains(binding) {
                uses.insert(*binding);
            }
        }
        MirTerminatorKind::Return(outputs) => {
            for output in outputs {
                collect_operand_read_with_defs(output, uses, defs);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            collect_operand_read_with_defs(future, uses, defs);
            if let Some(place) = result {
                collect_place_reads_with_defs(place, uses, defs);
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
}

fn collect_rvalue_reads_with_defs(
    value: &MirRvalue,
    uses: &mut HashSet<MirLocalId>,
    defs: &HashSet<MirLocalId>,
) {
    let read_operand = |operand: &MirOperand, uses: &mut HashSet<MirLocalId>| {
        collect_operand_read_with_defs(operand, uses, defs);
    };
    match value {
        MirRvalue::Use(operand) | MirRvalue::Unary(_, operand) | MirRvalue::Spawn(operand) => {
            read_operand(operand, uses);
        }
        MirRvalue::Binary(left, _, right) => {
            read_operand(left, uses);
            read_operand(right, uses);
        }
        MirRvalue::Range { start, step, end } => {
            read_operand(start, uses);
            if let Some(step) = step {
                read_operand(step, uses);
            }
            read_operand(end, uses);
        }
        MirRvalue::Call(call) => {
            for arg in &call.args {
                read_operand(arg.operand(), uses);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                read_operand(element, uses);
            }
        }
        MirRvalue::Index { base, indexing } => {
            read_operand(base, uses);
            collect_indexing_reads_with_defs(indexing, uses, defs);
        }
        MirRvalue::Member { base, .. } => read_operand(base, uses),
        MirRvalue::DynamicMember { base, member } => {
            read_operand(base, uses);
            read_operand(member, uses);
        }
        MirRvalue::Future { args, .. } => {
            for arg in args {
                read_operand(arg.operand(), uses);
            }
        }
        MirRvalue::MetaClass(_) | MirRvalue::Colon | MirRvalue::End => {}
    }
}

fn collect_place_reads_with_defs(
    place: &MirPlace,
    uses: &mut HashSet<MirLocalId>,
    defs: &HashSet<MirLocalId>,
) {
    match place {
        MirPlace::Local(_) | MirPlace::Binding(_) => {}
        MirPlace::Member(base, _) => collect_place_reads_with_defs(base, uses, defs),
        MirPlace::DynamicMember(base, member) => {
            collect_place_reads_with_defs(base, uses, defs);
            collect_operand_read_with_defs(member, uses, defs);
        }
        MirPlace::Index(base, indexing) => {
            collect_place_reads_with_defs(base, uses, defs);
            collect_indexing_reads_with_defs(indexing, uses, defs);
        }
    }
}

fn collect_indexing_reads_with_defs(
    indexing: &MirIndexing,
    uses: &mut HashSet<MirLocalId>,
    defs: &HashSet<MirLocalId>,
) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(operand) | MirIndexComponent::Logical(operand) => {
                collect_operand_read_with_defs(operand, uses, defs);
            }
            MirIndexComponent::Colon | MirIndexComponent::End { .. } => {}
        }
    }
}

fn collect_operand_read_with_defs(
    operand: &MirOperand,
    uses: &mut HashSet<MirLocalId>,
    defs: &HashSet<MirLocalId>,
) {
    if let MirOperand::Local(local) = operand {
        if !defs.contains(local) {
            uses.insert(*local);
        }
    }
}

fn place_root(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            place_root(base)
        }
        MirPlace::Binding(_) => None,
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
