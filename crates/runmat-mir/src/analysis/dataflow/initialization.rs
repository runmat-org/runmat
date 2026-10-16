use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    BasicBlockId, MirBody, MirLocalId, MirLocalKind, MirRvalue, MirStmtKind, MirTerminatorKind,
};
use runmat_types::ShapeFact;

use super::{mark_declared_storage, mark_place_assigned, simple_rvalue_fact, InitFact};

pub(super) struct InitDataflowResult {
    pub(super) in_states: Vec<Option<Vec<InitFact>>>,
}

pub(super) fn compute_init_dataflow(
    body: &MirBody,
    constructor_instance: Option<MirLocalId>,
) -> InitDataflowResult {
    let local_count = body.locals.len();
    if body.blocks.is_empty() {
        return InitDataflowResult {
            in_states: Vec::new(),
        };
    }

    let block_index: HashMap<BasicBlockId, usize> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    // A proven nonempty for-range must enter its body once before the loop's
    // own exit edge is feasible. Keep its reentry state separate from the
    // ordinary header join: joining the preheader with a backedge would lose
    // exactly that first-iteration fact.
    let reentry_sources: Vec<Option<HashSet<usize>>> = body
        .blocks
        .iter()
        .enumerate()
        .map(|(header_idx, block)| match &block.terminator.kind {
            MirTerminatorKind::For {
                iterable,
                body_block,
                exit_block,
                ..
            } if proven_nonempty_for_range(iterable) => Some(loop_reentry_sources(
                body,
                &block_index,
                header_idx,
                *body_block,
                *exit_block,
            )),
            _ => None,
        })
        .collect();
    let mut reentry_states: Vec<Option<Vec<InitFact>>> = vec![None; body.blocks.len()];

    let mut initial = vec![InitFact::Unassigned; local_count];
    for local in &body.locals {
        if matches!(local.kind, MirLocalKind::Parameter | MirLocalKind::Capture) {
            initial[local.id.0] = InitFact::DefinitelyAssigned;
        }
    }
    if let Some(local) = constructor_instance {
        initial[local.0] = InitFact::DefinitelyAssigned;
    }

    let mut in_states: Vec<Option<Vec<InitFact>>> = vec![None; body.blocks.len()];
    in_states[0] = Some(initial);

    let mut worklist = VecDeque::from([0usize]);
    while let Some(block_idx) = worklist.pop_front() {
        let Some(input) = in_states[block_idx].clone() else {
            continue;
        };
        let block = &body.blocks[block_idx];
        let output = transfer_block(block, input);

        for successor in successors(&block.terminator.kind) {
            let Some(&successor_idx) = block_index.get(&successor) else {
                continue;
            };
            let edge_output =
                if let MirTerminatorKind::For { exit_block, .. } = &block.terminator.kind {
                    if reentry_sources[block_idx].is_some() && successor == *exit_block {
                        // No zero-iteration exit. A break inside the body still
                        // reaches this successor through its own CFG edge.
                        let Some(reentry) = &reentry_states[block_idx] else {
                            continue;
                        };
                        transfer_block(block, reentry.clone())
                    } else {
                        output.clone()
                    }
                } else {
                    output.clone()
                };
            let reentry_changed = if reentry_sources[successor_idx]
                .as_ref()
                .is_some_and(|sources| sources.contains(&block_idx))
            {
                match &mut reentry_states[successor_idx] {
                    Some(existing) => join_into(existing, &edge_output),
                    slot @ None => {
                        *slot = Some(edge_output.clone());
                        true
                    }
                }
            } else {
                false
            };
            let changed = match &mut in_states[successor_idx] {
                Some(existing) => join_into(existing, &edge_output),
                slot @ None => {
                    *slot = Some(edge_output);
                    true
                }
            };
            if changed || reentry_changed {
                worklist.push_back(successor_idx);
            }
        }
    }

    InitDataflowResult { in_states }
}

fn proven_nonempty_for_range(iterable: &MirRvalue) -> bool {
    if !matches!(iterable, MirRvalue::Range { .. }) {
        return false;
    }
    matches!(
        simple_rvalue_fact(iterable, &[]).shape,
        ShapeFact::Shaped { ref dims }
            if matches!(dims.get(1), Some(runmat_types::DimensionFact::Known(length)) if *length > 0)
    )
}

fn loop_reentry_sources(
    body: &MirBody,
    block_index: &HashMap<BasicBlockId, usize>,
    header_idx: usize,
    body_block: BasicBlockId,
    exit_block: BasicBlockId,
) -> HashSet<usize> {
    // A natural-loop backedge comes from a block dominated by the header.
    // This also excludes an unrelated path that happens to reach the body's
    // continuation after leaving the loop.
    let mut reachable_without_header = HashSet::new();
    let mut outside_pending = vec![body.blocks[0].id];
    while let Some(id) = outside_pending.pop() {
        if id == body.blocks[header_idx].id {
            continue;
        }
        let Some(&idx) = block_index.get(&id) else {
            continue;
        };
        if reachable_without_header.insert(idx) {
            outside_pending.extend(successors(&body.blocks[idx].terminator.kind));
        }
    }
    let mut sources = HashSet::new();
    let mut seen = HashSet::new();
    let mut pending = vec![body_block];
    while let Some(id) = pending.pop() {
        if id == body.blocks[header_idx].id || id == exit_block {
            continue;
        }
        let Some(&idx) = block_index.get(&id) else {
            continue;
        };
        if !seen.insert(idx) {
            continue;
        }
        for successor in successors(&body.blocks[idx].terminator.kind) {
            if successor == body.blocks[header_idx].id {
                if !reachable_without_header.contains(&idx) {
                    sources.insert(idx);
                }
            } else if successor != exit_block {
                pending.push(successor);
            }
        }
    }
    sources
}

fn transfer_block(block: &crate::BasicBlock, mut state: Vec<InitFact>) -> Vec<InitFact> {
    for stmt in &block.statements {
        match &stmt.kind {
            MirStmtKind::Assign { place, .. } => mark_place_assigned(place, &mut state),
            MirStmtKind::MultiAssign { targets, .. } => {
                for target in &targets.targets {
                    match target {
                        crate::MirOutputTarget::Place(place) => {
                            mark_place_assigned(place, &mut state)
                        }
                        crate::MirOutputTarget::Sequence(target) => {
                            mark_place_assigned(target.base(), &mut state)
                        }
                        crate::MirOutputTarget::Discard => {}
                    }
                }
            }
            MirStmtKind::SequenceAssign { target, .. } => {
                let base = match target {
                    crate::MirSequenceTarget::Member { base, .. }
                    | crate::MirSequenceTarget::DynamicMember { base, .. }
                    | crate::MirSequenceTarget::CellContents { base, .. } => base,
                };
                mark_place_assigned(base, &mut state);
            }
            MirStmtKind::WorkspaceEffect { effect, bindings } => {
                mark_declared_storage(effect, bindings, &mut state);
            }
            MirStmtKind::CaptureSequence { .. } => {}
            MirStmtKind::Expr(_)
            | MirStmtKind::StatementOutput(_)
            | MirStmtKind::PlaceMutation(_)
            | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }

    match &block.terminator.kind {
        MirTerminatorKind::For { binding, .. } | MirTerminatorKind::ParFor { binding, .. } => {
            state[binding.0] = InitFact::DefinitelyAssigned;
        }
        MirTerminatorKind::Await {
            result: Some(place),
            ..
        } => {
            mark_place_assigned(place, &mut state);
        }
        MirTerminatorKind::Await { result: None, .. } => {}
        _ => {}
    }
    state
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
        MirTerminatorKind::ParFor {
            body_block,
            exit_block,
            ..
        }
        | MirTerminatorKind::Spmd {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            ..
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await { resume, .. } => vec![*resume],
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
