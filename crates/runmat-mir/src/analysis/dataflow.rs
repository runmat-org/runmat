use crate::{
    BasicBlockId, MirBody, MirLocalId, MirLocalKind, MirPlace, MirStmtKind, MirTerminatorKind,
};
use runmat_hir::{ShapeFact, TypeFact, ValueFlowFact};
use std::collections::{HashMap, VecDeque};

use super::{AnalysisStore, InitFact, MirLocalFact};

pub fn analyze_body(body: &MirBody, store: &mut AnalysisStore) {
    let local_count = body.locals.len();
    if body.blocks.is_empty() {
        return;
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
    let final_state = final_state.unwrap_or_else(|| vec![InitFact::Unassigned; local_count]);

    for local in &body.locals {
        store.mir_locals.insert(
            local.id,
            MirLocalFact {
                ty: TypeFact::Unknown,
                shape: ShapeFact::Unknown,
                value_flow: ValueFlowFact::UnknownList,
                initialized: final_state[local.id.0],
            },
        );
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
            MirStmtKind::Expr(_) | MirStmtKind::PlaceMutation(_) => {}
        }
    }

    if let MirTerminatorKind::For { binding, .. } = block.terminator.kind {
        state[binding.0] = InitFact::DefinitelyAssigned;
    }
    state
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
